use neuralnet::core::brain::MultiModalNeuralNetwork;
use neuralnet::core::nodenet::NodeMetadata;
use neuralnet::dendrites::text_dendrite::DendriteType;
use neuralnet::helpers::multimodal_controller::MultiModalInput;
use neuralnet::rag::{Generator, RagChunk, RagContext, RagPipeline, Retriever};
use neuralnet::training::trainer::TrainerBridgeTarget;

struct StaticRetriever {
    chunks: Vec<RagChunk>,
}

impl Retriever for StaticRetriever {
    fn retrieve(&self, _query: &str, top_k: usize) -> Vec<RagChunk> {
        self.chunks.iter().take(top_k).cloned().collect()
    }
}

struct EchoGenerator;

impl Generator for EchoGenerator {
    fn generate(&self, question: &str, context: &RagContext) -> String {
        format!(
            "RAG answer for '{}' using {} chunk(s)",
            question,
            context.chunks.len()
        )
    }
}

pub fn run_rag_walkthrough() {
    println!("\nRAG walkthrough");

    let retriever = StaticRetriever {
        chunks: vec![
            RagChunk {
                document_id: "doc_cat".to_string(),
                chunk_id: "doc_cat_0".to_string(),
                text: "cat on mat is a known training phrase".to_string(),
                score: 0.95,
            },
            RagChunk {
                document_id: "doc_vision".to_string(),
                chunk_id: "doc_vision_0".to_string(),
                text: "vision tokens can map to category hints".to_string(),
                score: 0.79,
            },
        ],
    };

    let pipeline = RagPipeline::new(retriever, EchoGenerator, 2);

    println!("  step 1: run plain text RAG answer");
    let plain = pipeline.answer_text("cat on mat");
    println!(
        "    query='{}' chunks={} answer='{}'",
        plain.context.query,
        plain.context.chunks.len(),
        plain.answer
    );

    println!("  step 2: train brain and request RAG answer with brain hint");
    let mut brain = MultiModalNeuralNetwork::new_multimodal();
    let metadata = NodeMetadata::with_lang("en");

    brain.train_labeled_pattern(
        "animal_cat",
        &MultiModalInput::Text("cat on mat".to_string()),
        &metadata,
        DendriteType::Statement,
        TrainerBridgeTarget::Cognitive,
    );

    let enriched = pipeline.answer_with_brain(
        &brain,
        &MultiModalInput::Text("cat on mat".to_string()),
    );

    println!(
        "    hint_label={:?} hint_score={:?}",
        enriched.model_label_hint,
        enriched.model_score_hint
    );
    println!(
        "    enriched answer='{}' with {} retrieved chunk(s)",
        enriched.answer,
        enriched.context.chunks.len()
    );

    println!("  step 3: feature-token input path");
    let feature_input = MultiModalInput::FeatureTokens {
        modality: "vision".to_string(),
        tokens: vec!["edge:04".to_string(), "shape:09".to_string()],
    };
    let feature_answer = pipeline.answer_with_brain(&brain, &feature_input);
    println!(
        "    feature query='{}' answer='{}'",
        feature_answer.context.query,
        feature_answer.answer
    );
}
