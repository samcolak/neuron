use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

#[derive(Debug)]
struct PendingSnapshotWrite {
    key: String,
    path: PathBuf,
    bytes: Vec<u8>,
    generation: u64,
}

#[derive(Debug, Default)]
struct SnapshotWriteState {
    next_generation: u64,
    pending_by_key: HashMap<String, PendingSnapshotWrite>,
    results_by_key: HashMap<String, BTreeMap<u64, Option<String>>>,
}

#[derive(Debug, Clone)]
struct SnapshotWriteWorker {
    state: Arc<(Mutex<SnapshotWriteState>, Condvar)>,
}

static SNAPSHOT_WRITE_WORKER: OnceLock<SnapshotWriteWorker> = OnceLock::new();

impl SnapshotWriteWorker {

    fn new() -> Self {
        let state = Arc::new((
            Mutex::new(SnapshotWriteState::default()),
            Condvar::new(),
        ));
        let thread_state = Arc::clone(&state);

        std::thread::Builder::new()
            .name("neuralnet-snapshot-writer".to_string())
            .spawn(move || run_snapshot_write_worker(thread_state))
            .expect("snapshot writer thread should start");

        Self { state }

    }

    fn submit(&self, key: String, path: PathBuf, bytes: Vec<u8>) -> u64 {

        let (lock, condvar) = &*self.state;
        let mut state = lock
            .lock()
            .expect("snapshot writer state lock should not be poisoned");

        state.next_generation = state.next_generation.saturating_add(1);
        let generation = state.next_generation;

        state.pending_by_key.insert(
            key.clone(),
            PendingSnapshotWrite {
                key,
                path,
                bytes,
                generation,
            },
        );

        condvar.notify_one();
        generation

    }

    fn wait_for_at_least(&self, key: &str, target_generation: u64) -> io::Result<()> {

        let (lock, condvar) = &*self.state;
        let mut state = lock
            .lock()
            .expect("snapshot writer state lock should not be poisoned");

        loop {
            if let Some(results) = state.results_by_key.get(key)
                && let Some((_, maybe_error)) = results.range(target_generation..).next()
            {
                if let Some(message) = maybe_error {
                    return Err(io::Error::other(message.clone()));
                }
                return Ok(());
            }

            state = condvar
                .wait(state)
                .expect("snapshot writer state lock should not be poisoned");
        }

    }

    fn latest_error(&self, key: &str) -> Option<String> {
        
        let (lock, _) = &*self.state;
        let state = lock
            .lock()
            .expect("snapshot writer state lock should not be poisoned");
        let results = state.results_by_key.get(key)?;
        let (_, maybe_error) = results.last_key_value()?;
        
        maybe_error.clone()

    }

    fn is_complete_at_least(&self, key: &str, target_generation: u64) -> bool {

        let (lock, _) = &*self.state;
        let state = lock
            .lock()
            .expect("snapshot writer state lock should not be poisoned");

        let Some(results) = state.results_by_key.get(key) else {
            return false;
        };

        results.range(target_generation..).next().is_some()
    }

}

fn worker() -> &'static SnapshotWriteWorker {
    SNAPSHOT_WRITE_WORKER.get_or_init(SnapshotWriteWorker::new)
}

fn run_snapshot_write_worker(state: Arc<(Mutex<SnapshotWriteState>, Condvar)>) {
    loop {
        let jobs = {
            let (lock, condvar) = &*state;
            let mut guard = lock
                .lock()
                .expect("snapshot writer state lock should not be poisoned");

            while guard.pending_by_key.is_empty() {
                guard = condvar
                    .wait(guard)
                    .expect("snapshot writer state lock should not be poisoned");
            }

            guard
                .pending_by_key
                .drain()
                .map(|(_, job)| job)
                .collect::<Vec<_>>()
        };

        for job in jobs {
            let write_result = write_snapshot_bytes_to_path(job.path.as_path(), job.bytes.as_slice());
            let error_message = write_result.err().map(|err| err.to_string());

            let (lock, condvar) = &*state;
            let mut guard = lock
                .lock()
                .expect("snapshot writer state lock should not be poisoned");
            let results = guard
                .results_by_key
                .entry(job.key)
                .or_insert_with(BTreeMap::new);
            results.insert(job.generation, error_message);

            while results.len() > 32 {
                if let Some((&oldest, _)) = results.first_key_value() {
                    results.remove(&oldest);
                }
            }

            condvar.notify_all();
        }
    }
}

pub(crate) fn submit_snapshot_write(key: String, path: PathBuf, bytes: Vec<u8>) -> u64 {
    worker().submit(key, path, bytes)
}

pub(crate) fn wait_for_snapshot_write(key: &str, target_generation: u64) -> io::Result<()> {
    worker().wait_for_at_least(key, target_generation)
}

pub(crate) fn latest_snapshot_write_error(key: &str) -> Option<String> {
    worker().latest_error(key)
}

pub(crate) fn is_snapshot_write_complete(key: &str, target_generation: u64) -> bool {
    worker().is_complete_at_least(key, target_generation)
}

pub(crate) fn write_snapshot_bytes_to_path(path: &Path, bytes: &[u8]) -> io::Result<()> {

    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let tmp_path = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("nrn")
    ));

    fs::write(&tmp_path, bytes)?;
    fs::rename(&tmp_path, path)?;

    Ok(())
    
}
