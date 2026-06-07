use std::fmt::{Display, Formatter};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::tensor::tensor4d::{Tensor4D, TensorError};

const TENSOR4D_MAGIC: [u8; 4] = *b"T4D1";
const HEADER_LEN: usize = 4 + 8 + 8 + 8 + 8;

#[derive(Debug)]
pub enum TensorIoError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Tensor(TensorError),
    InvalidMagic,
    CorruptPayload {
        expected_bytes: usize,
        actual_bytes: usize,
    },
}

impl Display for TensorIoError {

    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "tensor I/O error: {err}"),
            Self::Json(err) => write!(f, "tensor JSON error: {err}"),
            Self::Tensor(err) => write!(f, "tensor error: {err}"),
            Self::InvalidMagic => write!(f, "invalid tensor binary magic header"),
            Self::CorruptPayload {
                expected_bytes,
                actual_bytes,
            } => write!(
                f,
                "corrupt tensor payload: expected {} bytes, got {} bytes",
                expected_bytes, actual_bytes
            ),
        }
    }

}

impl std::error::Error for TensorIoError {}

impl From<std::io::Error> for TensorIoError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for TensorIoError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<TensorError> for TensorIoError {
    fn from(value: TensorError) -> Self {
        Self::Tensor(value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Tensor4DRecord {
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    data: Vec<f32>,
}

pub fn tensor4d_to_binary_bytes(tensor: &Tensor4D) -> Vec<u8> {
    let (n, c, h, w) = tensor.shape();

    let mut bytes = Vec::with_capacity(HEADER_LEN + tensor.len().saturating_mul(4));
    bytes.extend_from_slice(&TENSOR4D_MAGIC);
    bytes.extend_from_slice(&(n as u64).to_le_bytes());
    bytes.extend_from_slice(&(c as u64).to_le_bytes());
    bytes.extend_from_slice(&(h as u64).to_le_bytes());
    bytes.extend_from_slice(&(w as u64).to_le_bytes());

    for value in tensor.as_slice() {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    bytes
}

pub fn tensor4d_from_binary_bytes(bytes: &[u8]) -> Result<Tensor4D, TensorIoError> {
    if bytes.len() < HEADER_LEN {
        return Err(TensorIoError::CorruptPayload {
            expected_bytes: HEADER_LEN,
            actual_bytes: bytes.len(),
        });
    }

    if bytes[0..4] != TENSOR4D_MAGIC {
        return Err(TensorIoError::InvalidMagic);
    }

    let n = u64::from_le_bytes(bytes[4..12].try_into().unwrap_or([0; 8])) as usize;
    let c = u64::from_le_bytes(bytes[12..20].try_into().unwrap_or([0; 8])) as usize;
    let h = u64::from_le_bytes(bytes[20..28].try_into().unwrap_or([0; 8])) as usize;
    let w = u64::from_le_bytes(bytes[28..36].try_into().unwrap_or([0; 8])) as usize;

    let payload = &bytes[HEADER_LEN..];
    let expected_floats = n
        .checked_mul(c)
        .and_then(|x| x.checked_mul(h))
        .and_then(|x| x.checked_mul(w))
        .unwrap_or(usize::MAX);
    let expected_bytes = expected_floats.saturating_mul(4);

    if payload.len() != expected_bytes {
        return Err(TensorIoError::CorruptPayload {
            expected_bytes,
            actual_bytes: payload.len(),
        });
    }

    let mut data = Vec::with_capacity(expected_floats);
    for chunk in payload.chunks_exact(4) {
        let bytes4: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
        data.push(f32::from_le_bytes(bytes4));
    }

    Tensor4D::from_vec(n, c, h, w, data).map_err(Into::into)
}

pub fn save_tensor4d_binary(path: &Path, tensor: &Tensor4D) -> Result<(), TensorIoError> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let bytes = tensor4d_to_binary_bytes(tensor);
    let tmp_path = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("tbin")
    ));

    fs::write(&tmp_path, bytes)?;
    fs::rename(&tmp_path, path)?;
    Ok(())
}

pub fn load_tensor4d_binary(path: &Path) -> Result<Tensor4D, TensorIoError> {
    let bytes = fs::read(path)?;
    tensor4d_from_binary_bytes(bytes.as_slice())
}

pub fn save_tensor4d_json(path: &Path, tensor: &Tensor4D) -> Result<(), TensorIoError> {

    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let (n, c, h, w) = tensor.shape();
    let record = Tensor4DRecord {
        n,
        c,
        h,
        w,
        data: tensor.as_slice().to_vec(),
    };

    let encoded = serde_json::to_vec(&record)?;
    let tmp_path = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("json")
    ));

    fs::write(&tmp_path, encoded)?;
    fs::rename(&tmp_path, path)?;

    Ok(())

}

pub fn load_tensor4d_json(path: &Path) -> Result<Tensor4D, TensorIoError> {
    let bytes = fs::read(path)?;
    let record: Tensor4DRecord = serde_json::from_slice(bytes.as_slice())?;
    Tensor4D::from_vec(record.n, record.c, record.h, record.w, record.data).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    
    use super::*;
    use std::path::PathBuf;

    fn tensor_io_test_dir(test_name: &str) -> PathBuf {
        let mut path = PathBuf::from("./target/tensor_io_tests");
        path.push(test_name);
        path
    }

    fn cleanup_tensor_io_test_dir(test_name: &str) {
        let _ = fs::remove_dir_all(tensor_io_test_dir(test_name));
    }

    #[test]
    fn tensor_binary_round_trip_restores_shape_and_data() {
        let original = Tensor4D::from_vec(1, 1, 2, 2, vec![1.0, 2.0, 3.0, 4.0])
            .unwrap_or_else(|_| panic!("tensor should be valid"));

        let bytes = tensor4d_to_binary_bytes(&original);
        let restored = tensor4d_from_binary_bytes(bytes.as_slice())
            .unwrap_or_else(|_| panic!("binary roundtrip should succeed"));

        assert_eq!(restored.shape(), (1, 1, 2, 2));
        assert_eq!(restored.as_slice(), original.as_slice());
    }

    #[test]
    fn tensor_binary_reader_rejects_invalid_magic() {
        let mut bytes = vec![0u8; HEADER_LEN];
        bytes[0..4].copy_from_slice(b"BAD!");

        let result = tensor4d_from_binary_bytes(bytes.as_slice());
        assert!(matches!(result, Err(TensorIoError::InvalidMagic)));
    }

    #[test]
    fn tensor_json_file_round_trip_restores_tensor() {
        let test_name = "json_roundtrip";
        cleanup_tensor_io_test_dir(test_name);

        let dir = tensor_io_test_dir(test_name);
        let path = dir.join("tensor.json");

        let original = Tensor4D::from_vec(1, 1, 2, 2, vec![0.5, 1.5, 2.5, 3.5])
            .unwrap_or_else(|_| panic!("tensor should be valid"));

        assert!(save_tensor4d_json(path.as_path(), &original).is_ok());

        let restored = load_tensor4d_json(path.as_path())
            .unwrap_or_else(|_| panic!("json roundtrip should succeed"));

        assert_eq!(restored.shape(), original.shape());
        assert_eq!(restored.as_slice(), original.as_slice());

        cleanup_tensor_io_test_dir(test_name);
    }

    #[test]
    fn tensor_binary_file_round_trip_restores_tensor() {
        let test_name = "binary_roundtrip";
        cleanup_tensor_io_test_dir(test_name);

        let dir = tensor_io_test_dir(test_name);
        let path = dir.join("tensor.tbin");

        let original = Tensor4D::from_vec(2, 1, 1, 2, vec![9.0, 8.0, 7.0, 6.0])
            .unwrap_or_else(|_| panic!("tensor should be valid"));

        assert!(save_tensor4d_binary(path.as_path(), &original).is_ok());

        let restored = load_tensor4d_binary(path.as_path())
            .unwrap_or_else(|_| panic!("binary file roundtrip should succeed"));

        assert_eq!(restored.shape(), original.shape());
        assert_eq!(restored.as_slice(), original.as_slice());

        cleanup_tensor_io_test_dir(test_name);
    }
}
