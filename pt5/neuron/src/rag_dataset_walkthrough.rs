use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use neuralnet::rag::{import_csv, RagCsvImportConfig, RagDocument};

fn dataset_path() -> PathBuf {

    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("demodata")
        .join("synthetic_knowledge_items.csv")

    }

fn dump_path() -> PathBuf {

    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("rag_dataset_ingestion_dump.txt")

}

fn validate_documents(documents: &[RagDocument]) -> Result<(), String> {

    if documents.is_empty() {
        return Err("no documents were imported from CSV".to_string());
    }

    for (idx, doc) in documents.iter().enumerate() {
        
        if doc.text.trim().is_empty() {
            return Err(format!("document at index {} has empty ki_text", idx));
        }

        if !doc.metadata.contains_key("ki_topic") {
            return Err(format!("document at index {} is missing ki_topic metadata", idx));
        }

        if !doc.metadata.contains_key("alt_ki_text") {
            return Err(format!("document at index {} is missing alt_ki_text metadata", idx));
        }

        if !doc.metadata.contains_key("bad_ki_text") {
            return Err(format!("document at index {} is missing bad_ki_text metadata", idx));
        }

    }

    Ok(())

}

fn build_dump(documents: &[RagDocument]) -> String {
    
    let mut lines = Vec::new();
    lines.push("RAG dataset ingestion dump".to_string());
    lines.push(format!("documents_imported={}", documents.len()));

    let mut topic_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut metadata_key_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut metadata_value_non_empty_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut duplicate_ids: BTreeMap<String, usize> = BTreeMap::new();
    let mut seen_ids: BTreeMap<String, usize> = BTreeMap::new();

    let mut total_text_chars: usize = 0;
    let mut min_text_chars: usize = usize::MAX;
    let mut max_text_chars: usize = 0;
    let mut empty_text_count: usize = 0;

    let mut total_alt_chars: usize = 0;
    let mut total_bad_chars: usize = 0;
    let mut present_alt_count: usize = 0;
    let mut present_bad_count: usize = 0;

    for doc in documents {
        if let Some(topic) = doc.metadata.get("ki_topic") {
            *topic_counts.entry(topic.clone()).or_insert(0) += 1;
        }

        for (k, v) in &doc.metadata {
            *metadata_key_counts.entry(k.clone()).or_insert(0) += 1;
            if !v.trim().is_empty() {
                *metadata_value_non_empty_counts.entry(k.clone()).or_insert(0) += 1;
            }
        }

        let text_len = doc.text.chars().count();
        total_text_chars += text_len;
        min_text_chars = min_text_chars.min(text_len);
        max_text_chars = max_text_chars.max(text_len);
        if doc.text.trim().is_empty() {
            empty_text_count += 1;
        }

        if let Some(alt_text) = doc.metadata.get("alt_ki_text")
            && !alt_text.trim().is_empty()
        {
            present_alt_count += 1;
            total_alt_chars += alt_text.chars().count();
        }

        if let Some(bad_text) = doc.metadata.get("bad_ki_text")
            && !bad_text.trim().is_empty()
        {
            present_bad_count += 1;
            total_bad_chars += bad_text.chars().count();
        }

        let seen = seen_ids.entry(doc.id.clone()).or_insert(0);
        *seen += 1;
        if *seen > 1 {
            duplicate_ids.insert(doc.id.clone(), *seen);
        }
    }

    if documents.is_empty() {
        min_text_chars = 0;
    }

    let avg_text_chars = if documents.is_empty() {
        0.0
    } else {
        total_text_chars as f64 / documents.len() as f64
    };

    let avg_alt_chars = if present_alt_count == 0 {
        0.0
    } else {
        total_alt_chars as f64 / present_alt_count as f64
    };

    let avg_bad_chars = if present_bad_count == 0 {
        0.0
    } else {
        total_bad_chars as f64 / present_bad_count as f64
    };

    lines.push(format!("distinct_topics={}", topic_counts.len()));
    lines.push(format!("text_chars_total={}", total_text_chars));
    lines.push(format!("text_chars_avg={:.2}", avg_text_chars));
    lines.push(format!("text_chars_min={}", min_text_chars));
    lines.push(format!("text_chars_max={}", max_text_chars));
    lines.push(format!("empty_text_documents={}", empty_text_count));
    lines.push(format!("alt_text_present_documents={}", present_alt_count));
    lines.push(format!("alt_text_chars_avg={:.2}", avg_alt_chars));
    lines.push(format!("bad_text_present_documents={}", present_bad_count));
    lines.push(format!("bad_text_chars_avg={:.2}", avg_bad_chars));

    lines.push("duplicate_document_ids:".to_string());

    if duplicate_ids.is_empty() {
        lines.push("  none".to_string());
    } else {
        for (id, count) in duplicate_ids {
            lines.push(format!("  id={} occurrences={}", id, count));
        }
    }

    lines.push("metadata_key_coverage:".to_string());
    for (key, count) in &metadata_key_counts {
        let non_empty = metadata_value_non_empty_counts.get(key).copied().unwrap_or(0);
        lines.push(format!(
            "  key='{}' present_in={} non_empty_values={}",
            key, count, non_empty
        ));
    }

    lines.push("topic_distribution:".to_string());
    for (topic, count) in &topic_counts {
        lines.push(format!("  topic='{}' count={}", topic, count));
    }

    lines.push("documents_verbose:".to_string());

    for (idx, doc) in documents.iter().enumerate() {
        let topic = doc
            .metadata
            .get("ki_topic")
            .map(String::as_str)
            .unwrap_or("<missing>");
        let alt_text = doc
            .metadata
            .get("alt_ki_text")
            .map(String::as_str)
            .unwrap_or("");
        let bad_text = doc
            .metadata
            .get("bad_ki_text")
            .map(String::as_str)
            .unwrap_or("");

        let normalize = |s: &str| s.replace(['\n', '\r'], " ");

        lines.push(format!("  [{}]", idx));
        lines.push(format!("    id='{}'", doc.id));
        lines.push(format!("    topic='{}'", topic));
        lines.push(format!("    text_chars={}", doc.text.chars().count()));
        lines.push(format!("    alt_text_chars={}", alt_text.chars().count()));
        lines.push(format!("    bad_text_chars={}", bad_text.chars().count()));
        lines.push(format!("    text='{}'", normalize(&doc.text)));
        lines.push(format!("    alt_text='{}'", normalize(alt_text)));
        lines.push(format!("    bad_text='{}'", normalize(bad_text)));

        let mut extra_metadata: Vec<_> = doc
            .metadata
            .iter()
            .filter(|(k, _)| {
                k.as_str() != "ki_topic" && k.as_str() != "alt_ki_text" && k.as_str() != "bad_ki_text"
            })
            .collect();
        extra_metadata.sort_by(|a, b| a.0.cmp(b.0));

        if extra_metadata.is_empty() {
            lines.push("    extra_metadata=none".to_string());
        } else {
            lines.push("    extra_metadata:".to_string());
            for (k, v) in extra_metadata {
                lines.push(format!("      {}='{}'", k, normalize(v)));
            }
        }
    }

    lines.join("\n")

}

pub fn run_rag_dataset_walkthrough() {

    println!("\nRAG dataset walkthrough");

    let csv_path = dataset_path();
    println!("  step 1: import CSV at {}", csv_path.display());

    let config = RagCsvImportConfig {
        text_column: "ki_text".to_string(),
        id_column: None,
        metadata_columns: vec![
            "ki_topic".to_string(),
            "alt_ki_text".to_string(),
            "bad_ki_text".to_string(),
        ],
    };

    let documents = match import_csv(csv_path.as_path(), &config) {
        Ok(docs) => docs,
        Err(err) => {
            println!("    import failed: {}", err);
            return;
        }
    };

    println!("    imported {} row-level document(s)", documents.len());

    println!("  step 2: validate imported content");
    match validate_documents(documents.as_slice()) {
        Ok(()) => println!("    validation passed"),
        Err(reason) => {
            println!("    validation failed: {}", reason);
            return;
        }
    }

    println!("  step 3: create ingestion dump");
    let dump = build_dump(documents.as_slice());
    let out_path = dump_path();

    if let Some(parent) = out_path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    match fs::write(out_path.as_path(), dump) {
        Ok(()) => println!("    wrote dump to {}", out_path.display()),
        Err(err) => println!("    failed to write dump: {}", err),
    }
    
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_import_and_validation_succeeds_for_demo_csv() {
        let config = RagCsvImportConfig {
            text_column: "ki_text".to_string(),
            id_column: None,
            metadata_columns: vec![
                "ki_topic".to_string(),
                "alt_ki_text".to_string(),
                "bad_ki_text".to_string(),
            ],
        };

        let docs = import_csv(dataset_path().as_path(), &config)
            .unwrap_or_else(|_| panic!("demo csv import should succeed"));

        validate_documents(docs.as_slice())
            .unwrap_or_else(|_| panic!("demo csv validation should succeed"));

        assert!(!docs.is_empty());
    }
}
