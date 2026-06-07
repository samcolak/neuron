use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RagDocument {
    pub id: String,
    pub text: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RagChunk {
    pub document_id: String,
    pub chunk_id: String,
    pub text: String,
    pub score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RagContext {
    pub query: String,
    pub chunks: Vec<RagChunk>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RagAnswer {
    pub answer: String,
    pub context: RagContext,
    pub model_label_hint: Option<String>,
    pub model_score_hint: Option<f64>,
}
