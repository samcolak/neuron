use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use crate::core::nodenet::NodeMetadata;
use crate::dendrites::text_dendrite::DendriteType;
use crate::helpers::multimodal_controller::MultiModalInput;
use serde::{Deserialize, Serialize};

const TRANSACTION_LOG_MAGIC_V1: [u8; 4] = *b"NTL1";
const MAX_RECORD_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum TransactionTarget {
    Cognitive,
    Memory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum TransactionOperation {
    Upsert {
        target: TransactionTarget,
        content: MultiModalInput,
        metadata: NodeMetadata,
        dendrite_type: DendriteType,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TransactionRecord {
    operation: TransactionOperation,
}

#[derive(Debug)]
struct TransactionWriteJob {
    key: String,
    path: PathBuf,
    record: TransactionRecord,
    sequence: u64,
}

#[derive(Debug, Default)]
struct TransactionLogState {
    next_sequence: u64,
    pending_by_key: HashMap<String, Vec<TransactionWriteJob>>,
    results_by_key: HashMap<String, BTreeMap<u64, Option<String>>>,
}

#[derive(Debug, Clone)]
struct TransactionLogWorker {
    state: Arc<(Mutex<TransactionLogState>, Condvar)>,
}

static TRANSACTION_LOG_WORKER: OnceLock<TransactionLogWorker> = OnceLock::new();
static WAL_FSYNC_ENABLED: OnceLock<bool> = OnceLock::new();

fn wal_fsync_enabled() -> bool {
    *WAL_FSYNC_ENABLED.get_or_init(|| {
        env::var("NEURALNET_WAL_FSYNC")
            .ok()
            .map(|value| {
                let normalized = value.trim().to_ascii_lowercase();
                normalized == "1" || normalized == "true" || normalized == "yes"
            })
            .unwrap_or(false)
    })
}

impl TransactionLogWorker {
    fn new() -> Self {
        let state = Arc::new((
            Mutex::new(TransactionLogState::default()),
            Condvar::new(),
        ));
        let thread_state = Arc::clone(&state);

        std::thread::Builder::new()
            .name("neuralnet-transaction-log-writer".to_string())
            .spawn(move || run_transaction_log_worker(thread_state))
            .expect("transaction log writer thread should start");

        Self { state }
    }

    fn submit(&self, key: String, path: PathBuf, operation: TransactionOperation) -> u64 {
        let (lock, condvar) = &*self.state;
        let mut state = lock
            .lock()
            .expect("transaction log state lock should not be poisoned");

        state.next_sequence = state.next_sequence.saturating_add(1);
        let sequence = state.next_sequence;

        let record = TransactionRecord { operation };
        state
            .pending_by_key
            .entry(key.clone())
            .or_default()
            .push(TransactionWriteJob {
                key,
                path,
                record,
                sequence,
            });

        condvar.notify_one();
        sequence
    }

    fn wait_for_at_least(&self, key: &str, target_sequence: u64) -> io::Result<()> {
        let (lock, condvar) = &*self.state;
        let mut state = lock
            .lock()
            .expect("transaction log state lock should not be poisoned");

        loop {
            if let Some(results) = state.results_by_key.get(key)
                && let Some((_, maybe_error)) = results.range(target_sequence..).next()
            {
                if let Some(message) = maybe_error {
                    return Err(io::Error::other(message.clone()));
                }
                return Ok(());
            }

            state = condvar
                .wait(state)
                .expect("transaction log state lock should not be poisoned");
        }
    }

    fn latest_error(&self, key: &str) -> Option<String> {
        let (lock, _) = &*self.state;
        let state = lock
            .lock()
            .expect("transaction log state lock should not be poisoned");

        let results = state.results_by_key.get(key)?;
        let (_, maybe_error) = results.last_key_value()?;
        maybe_error.clone()
    }
}

fn worker() -> &'static TransactionLogWorker {
    TRANSACTION_LOG_WORKER.get_or_init(TransactionLogWorker::new)
}

fn run_transaction_log_worker(state: Arc<(Mutex<TransactionLogState>, Condvar)>) {
    loop {
        let jobs = {
            let (lock, condvar) = &*state;
            let mut guard = lock
                .lock()
                .expect("transaction log state lock should not be poisoned");

            while guard.pending_by_key.is_empty() {
                guard = condvar
                    .wait(guard)
                    .expect("transaction log state lock should not be poisoned");
            }

            let mut collected = Vec::new();
            for (_key, mut pending) in guard.pending_by_key.drain() {
                collected.append(&mut pending);
            }
            collected
        };

        for job in jobs {
            let write_result = append_record_to_log(job.path.as_path(), &job.record);
            let error_message = write_result.err().map(|err| err.to_string());

            let (lock, condvar) = &*state;
            let mut guard = lock
                .lock()
                .expect("transaction log state lock should not be poisoned");

            let results = guard
                .results_by_key
                .entry(job.key)
                .or_insert_with(BTreeMap::new);
            results.insert(job.sequence, error_message);

            while results.len() > 128 {
                if let Some((&oldest, _)) = results.first_key_value() {
                    results.remove(&oldest);
                }
            }

            condvar.notify_all();
        }
    }
}

pub(crate) fn append_transaction(
    key: String,
    path: PathBuf,
    operation: TransactionOperation,
) -> u64 {
    worker().submit(key, path, operation)
}

pub(crate) fn flush_transactions(key: &str, target_sequence: u64) -> io::Result<()> {
    worker().wait_for_at_least(key, target_sequence)
}

pub(crate) fn latest_transaction_error(key: &str) -> Option<String> {
    worker().latest_error(key)
}

pub(crate) fn truncate_transaction_log(path: &Path) -> io::Result<()> {
    rewrite_transaction_log(path, &[])
}

pub(crate) fn load_and_sanitize_transaction_log(path: &Path) -> io::Result<Vec<TransactionOperation>> {
    let mut file = match OpenOptions::new().read(true).open(path) {
        Ok(file) => file,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err),
    };

    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    if bytes.is_empty() {
        return Ok(Vec::new());
    }

    if bytes.len() < TRANSACTION_LOG_MAGIC_V1.len()
        || bytes[0..4] != TRANSACTION_LOG_MAGIC_V1
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "invalid transaction log header for '{}'",
                path.display()
            ),
        ));
    }

    let mut cursor = TRANSACTION_LOG_MAGIC_V1.len();
    let mut operations = Vec::new();

    while cursor + 4 <= bytes.len() {
        let len_bytes: [u8; 4] = bytes[cursor..cursor + 4]
            .try_into()
            .expect("slice of 4 bytes should convert");
        cursor += 4;

        let record_len = u32::from_le_bytes(len_bytes) as usize;
        if record_len == 0 || record_len > MAX_RECORD_BYTES {
            break;
        }

        if cursor + record_len > bytes.len() {
            break;
        }

        let record_bytes = &bytes[cursor..cursor + record_len];
        cursor += record_len;

        if let Ok(record) = bincode::deserialize::<TransactionRecord>(record_bytes) {
            operations.push(record.operation);
        }
    }

    rewrite_transaction_log(path, operations.as_slice())?;
    Ok(operations)
}

fn append_record_to_log(path: &Path, record: &TransactionRecord) -> io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let encoded = bincode::serialize(record).map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to encode transaction log record: {err}"),
        )
    })?;

    if encoded.len() > MAX_RECORD_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("transaction log record too large: {} bytes", encoded.len()),
        ));
    }

    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    let file_is_empty = file.metadata()?.len() == 0;
    if file_is_empty {
        file.write_all(&TRANSACTION_LOG_MAGIC_V1)?;
    }

    let len_bytes = (encoded.len() as u32).to_le_bytes();
    file.write_all(&len_bytes)?;
    file.write_all(encoded.as_slice())?;
    file.flush()?;
    if wal_fsync_enabled() {
        file.sync_data()?;
    }

    Ok(())
}

fn rewrite_transaction_log(path: &Path, operations: &[TransactionOperation]) -> io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let tmp_path = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("wal")
    ));

    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&tmp_path)?;

    file.write_all(&TRANSACTION_LOG_MAGIC_V1)?;

    for operation in operations {
        let record = TransactionRecord {
            operation: operation.clone(),
        };

        let encoded = bincode::serialize(&record).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to encode transaction log record: {err}"),
            )
        })?;

        let len_bytes = (encoded.len() as u32).to_le_bytes();
        file.write_all(&len_bytes)?;
        file.write_all(encoded.as_slice())?;
    }

    file.flush()?;
    if wal_fsync_enabled() {
        file.sync_data()?;
    }
    fs::rename(&tmp_path, path)?;

    Ok(())
}
