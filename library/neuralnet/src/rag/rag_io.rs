use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use super::types::RagDocument;

#[derive(Debug)]
pub enum RagIoError {
    Io(std::io::Error),
    Csv(csv::Error),
    InvalidPath(PathBuf),
    NonUtf8File(PathBuf),
    MissingCsvColumn {
        column: String,
        path: PathBuf,
    },
}

impl Display for RagIoError {

    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "rag I/O error: {}", err),
            Self::Csv(err) => write!(f, "rag CSV parse error: {}", err),
            Self::InvalidPath(path) => {
                write!(f, "invalid path for rag import: {}", path.display())
            }
            Self::NonUtf8File(path) => {
                write!(f, "rag importer only supports UTF-8 files: {}", path.display())
            }
            Self::MissingCsvColumn { column, path } => {
                write!(
                    f,
                    "required CSV column '{}' not found in {}",
                    column,
                    path.display()
                )
            }
        }
    }

}

impl Error for RagIoError {}

impl From<std::io::Error> for RagIoError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<csv::Error> for RagIoError {
    fn from(value: csv::Error) -> Self {
        Self::Csv(value)
    }
}

#[derive(Debug, Clone)]
pub struct RagFileImportConfig {
    pub include_extensions: Vec<String>,
    pub recursive: bool,
}

impl Default for RagFileImportConfig {
    fn default() -> Self {
        Self {
            include_extensions: vec!["txt".to_string(), "md".to_string()],
            recursive: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RagCsvImportConfig {
    pub text_column: String,
    pub id_column: Option<String>,
    pub metadata_columns: Vec<String>,
}

impl Default for RagCsvImportConfig {
    fn default() -> Self {
        Self {
            text_column: "ki_text".to_string(),
            id_column: None,
            metadata_columns: Vec::new(),
        }
    }
}

fn normalize_extensions(exts: &[String]) -> Vec<String> {
    exts.iter()
        .map(|item| item.trim().trim_start_matches('.').to_ascii_lowercase())
        .filter(|item| !item.is_empty())
        .collect()
}

fn should_import(path: &Path, normalized_extensions: &[String]) -> bool {
    if normalized_extensions.is_empty() {
        return true;
    }

    let Some(ext) = path.extension().and_then(|value| value.to_str()) else {
        return false;
    };

    let ext_lower = ext.to_ascii_lowercase();
    normalized_extensions.iter().any(|item| item == &ext_lower)
}

fn build_document_id(path: &Path, text: &str) -> String {
    let mut hasher = DefaultHasher::new();
    path.to_string_lossy().hash(&mut hasher);
    text.len().hash(&mut hasher);
    format!("doc_{:016x}", hasher.finish())
}

fn build_document_id_for_row(path: &Path, row_index: usize, text: &str) -> String {
    let mut hasher = DefaultHasher::new();
    path.to_string_lossy().hash(&mut hasher);
    row_index.hash(&mut hasher);
    text.len().hash(&mut hasher);
    format!("doc_{:016x}", hasher.finish())
}

pub fn import_csv(path: &Path, config: &RagCsvImportConfig) -> Result<Vec<RagDocument>, RagIoError> {
    if !path.exists() || !path.is_file() {
        return Err(RagIoError::InvalidPath(path.to_path_buf()));
    }

    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();

    let text_column = config.text_column.trim();
    let text_index = headers
        .iter()
        .position(|value| value == text_column)
        .ok_or_else(|| RagIoError::MissingCsvColumn {
            column: text_column.to_string(),
            path: path.to_path_buf(),
        })?;

    let id_index = if let Some(id_column) = &config.id_column {
        let col = id_column.trim();
        Some(
            headers
                .iter()
                .position(|value| value == col)
                .ok_or_else(|| RagIoError::MissingCsvColumn {
                    column: col.to_string(),
                    path: path.to_path_buf(),
                })?,
        )
    } else {
        None
    };

    let metadata_indices: Vec<(usize, String)> = if config.metadata_columns.is_empty() {
        headers
            .iter()
            .enumerate()
            .filter_map(|(idx, name)| {
                if idx == text_index || Some(idx) == id_index {
                    None
                } else {
                    Some((idx, name.to_string()))
                }
            })
            .collect()
    } else {
        let mut indices = Vec::new();
        for column in &config.metadata_columns {
            let col = column.trim();
            let idx = headers
                .iter()
                .position(|value| value == col)
                .ok_or_else(|| RagIoError::MissingCsvColumn {
                    column: col.to_string(),
                    path: path.to_path_buf(),
                })?;
            indices.push((idx, col.to_string()));
        }
        indices
    };

    let mut documents = Vec::new();
    for (row_index, row_result) in reader.records().enumerate() {
        let row = row_result?;
        let text = row
            .get(text_index)
            .map(str::trim)
            .unwrap_or_default()
            .to_string();

        if text.is_empty() {
            continue;
        }

        let mut metadata = HashMap::new();
        metadata.insert("source_path".to_string(), path.to_string_lossy().to_string());
        metadata.insert("row_index".to_string(), row_index.to_string());

        for (idx, key) in &metadata_indices {
            if let Some(value) = row.get(*idx) {
                let trimmed = value.trim();
                if !trimmed.is_empty() {
                    metadata.insert(key.clone(), trimmed.to_string());
                }
            }
        }

        let id = if let Some(idx) = id_index {
            let candidate = row.get(idx).map(str::trim).unwrap_or_default();
            if candidate.is_empty() {
                build_document_id_for_row(path, row_index, text.as_str())
            } else {
                candidate.to_string()
            }
        } else {
            build_document_id_for_row(path, row_index, text.as_str())
        };

        documents.push(RagDocument { id, text, metadata });
    }

    Ok(documents)
}

pub fn import_file(path: &Path) -> Result<RagDocument, RagIoError> {
    if !path.exists() || !path.is_file() {
        return Err(RagIoError::InvalidPath(path.to_path_buf()));
    }

    let bytes = fs::read(path)?;
    let text = String::from_utf8(bytes).map_err(|_| RagIoError::NonUtf8File(path.to_path_buf()))?;

    let mut metadata = HashMap::new();
    metadata.insert("source_path".to_string(), path.to_string_lossy().to_string());

    if let Some(name) = path.file_name().and_then(|value| value.to_str()) {
        metadata.insert("file_name".to_string(), name.to_string());
    }

    Ok(RagDocument {
        id: build_document_id(path, text.as_str()),
        text,
        metadata,
    })
}

pub fn import_directory(path: &Path, config: &RagFileImportConfig) -> Result<Vec<RagDocument>, RagIoError> {

    if !path.exists() || !path.is_dir() {
        return Err(RagIoError::InvalidPath(path.to_path_buf()));
    }

    let normalized_extensions = normalize_extensions(config.include_extensions.as_slice());
    let mut documents = Vec::new();

    import_directory_inner(
        path,
        config.recursive,
        normalized_extensions.as_slice(),
        &mut documents,
    )?;

    Ok(documents)

}

fn import_directory_inner(
    path: &Path,
    recursive: bool,
    normalized_extensions: &[String],
    documents: &mut Vec<RagDocument>,
) -> Result<(), RagIoError> {

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let entry_path = entry.path();

        if entry_path.is_dir() {
            if recursive {
                import_directory_inner(
                    entry_path.as_path(),
                    recursive,
                    normalized_extensions,
                    documents,
                )?;
            }
            continue;
        }

        if !should_import(entry_path.as_path(), normalized_extensions) {
            continue;
        }

        let doc = import_file(entry_path.as_path())?;
        documents.push(doc);
    }

    Ok(())

}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_path(suffix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!("neuralnet_rag_io_{}_{}", suffix, nanos))
    }

    #[test]
    fn import_file_reads_utf8_text_into_document() {
        let file_path = unique_temp_path("file.txt");
        fs::write(&file_path, "hello rag importer")
            .unwrap_or_else(|_| panic!("temp file should be writable"));

        let doc = import_file(file_path.as_path())
            .unwrap_or_else(|_| panic!("import file should succeed"));

        assert_eq!(doc.text, "hello rag importer");
        assert!(doc.id.starts_with("doc_"));
        assert_eq!(
            doc.metadata.get("source_path").map(String::as_str),
            Some(file_path.to_string_lossy().as_ref())
        );

        let _ = fs::remove_file(file_path);
    }

    #[test]
    fn import_directory_filters_by_extension() {
        let dir_path = unique_temp_path("dir");
        fs::create_dir_all(&dir_path)
            .unwrap_or_else(|_| panic!("temp dir should be created"));

        let txt_path = dir_path.join("a.txt");
        let md_path = dir_path.join("b.md");
        let json_path = dir_path.join("c.json");

        fs::write(&txt_path, "text doc")
            .unwrap_or_else(|_| panic!("txt should be writable"));
        fs::write(&md_path, "markdown doc")
            .unwrap_or_else(|_| panic!("md should be writable"));
        fs::write(&json_path, "{\"k\":\"v\"}")
            .unwrap_or_else(|_| panic!("json should be writable"));

        let config = RagFileImportConfig {
            include_extensions: vec!["txt".to_string()],
            recursive: false,
        };

        let docs = import_directory(dir_path.as_path(), &config)
            .unwrap_or_else(|_| panic!("directory import should succeed"));

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].metadata.get("file_name").map(String::as_str), Some("a.txt"));

        let _ = fs::remove_dir_all(dir_path);
    }

    #[test]
    fn import_csv_builds_row_level_documents() {
        let file_path = unique_temp_path("rows.csv");
        let csv_text = "ki_topic,ki_text,alt_ki_text\nTopic A,Primary A,Alt A\nTopic B,Primary B,Alt B\n";
        fs::write(&file_path, csv_text)
            .unwrap_or_else(|_| panic!("csv test file should be writable"));

        let docs = import_csv(file_path.as_path(), &RagCsvImportConfig::default())
            .unwrap_or_else(|_| panic!("csv import should succeed"));

        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].text, "Primary A");
        assert_eq!(docs[1].text, "Primary B");
        assert_eq!(
            docs[0].metadata.get("ki_topic").map(String::as_str),
            Some("Topic A")
        );

        let _ = fs::remove_file(file_path);
    }

    #[test]
    fn import_csv_supports_custom_text_column() {
        let file_path = unique_temp_path("custom.csv");
        let csv_text = "ki_topic,ki_text,alt_ki_text\nTopic A,Primary A,Alt A\n";
        fs::write(&file_path, csv_text)
            .unwrap_or_else(|_| panic!("csv test file should be writable"));

        let config = RagCsvImportConfig {
            text_column: "alt_ki_text".to_string(),
            id_column: None,
            metadata_columns: vec!["ki_topic".to_string()],
        };

        let docs = import_csv(file_path.as_path(), &config)
            .unwrap_or_else(|_| panic!("csv import should succeed"));

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].text, "Alt A");
        assert_eq!(
            docs[0].metadata.get("ki_topic").map(String::as_str),
            Some("Topic A")
        );

        let _ = fs::remove_file(file_path);
    }
    
}
