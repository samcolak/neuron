use neuralnet::core::brain::MultiModalNeuralNetwork;
use neuralnet::core::nodenet::NodeMetadata;
use neuralnet::dendrites::text_dendrite::DendriteType;
use neuralnet::helpers::multimodal_controller::MultiModalInput;
use neuralnet::rag::{Generator, RagChunk, RagContext, RagPipeline, Retriever};
use neuralnet::tensor::tensor4d::Tensor4D;
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
        format!("q={} chunks={}", question, context.chunks.len())
    }
}

#[test]
fn rag_pipeline_uses_brain_hint_and_context() {
    
    let retriever = StaticRetriever {
        chunks: vec![RagChunk {
            document_id: "doc_1".to_string(),
            chunk_id: "doc_1_0".to_string(),
            text: "cat on mat context".to_string(),
            score: 0.91,
        }],
    };

    let pipeline = RagPipeline::new(retriever, EchoGenerator, 3);

    let mut brain = MultiModalNeuralNetwork::new_multimodal();
    let metadata = NodeMetadata::with_lang("en");
    
    brain.train_labeled_pattern(
        "animal_cat",
        &MultiModalInput::Text("cat on mat".to_string()),
        &metadata,
        DendriteType::Statement,
        TrainerBridgeTarget::Cognitive,
    );

    let answer = pipeline.answer_with_brain(
        &brain,
        &MultiModalInput::Text("cat on mat".to_string()),
    );

    assert_eq!(answer.context.query, "cat on mat");
    assert_eq!(answer.context.chunks.len(), 1);
    assert_eq!(answer.model_label_hint, Some("animal_cat".to_string()));
    assert!(answer.model_score_hint.unwrap_or(0.0) >= 0.60);

}

#[test]
fn tensor_multilayer_matches_single_layer_when_extra_channel_is_zeroed() {
    let single = Tensor4D::from_vec(1, 1, 2, 2, vec![1.0, 2.0, 3.0, 4.0])
        .unwrap_or_else(|_| panic!("single tensor should be valid"));

    let multi = Tensor4D::from_vec(
        1,
        2,
        2,
        2,
        vec![
            1.0, 2.0, 3.0, 4.0, // channel 0
            0.0, 0.0, 0.0, 0.0, // channel 1
        ],
    )
    .unwrap_or_else(|_| panic!("multi tensor should be valid"));

    let kernel_single = Tensor4D::from_vec(1, 1, 1, 1, vec![2.0])
        .unwrap_or_else(|_| panic!("single kernel should be valid"));

    let kernel_multi = Tensor4D::from_vec(1, 2, 1, 1, vec![2.0, 0.0])
        .unwrap_or_else(|_| panic!("multi kernel should be valid"));

    let out_single = single
        .conv2d_valid(&kernel_single, None, 1, 1)
        .unwrap_or_else(|_| panic!("single conv should succeed"));
    let out_multi = multi
        .conv2d_valid(&kernel_multi, None, 1, 1)
        .unwrap_or_else(|_| panic!("multi conv should succeed"));

    assert_eq!(out_single.shape(), out_multi.shape());
    assert_eq!(out_single.as_slice(), out_multi.as_slice());
}

#[test]
fn tensor_multilayer_uses_additional_channel_signal() {
    let single = Tensor4D::from_vec(1, 1, 1, 1, vec![2.0])
        .unwrap_or_else(|_| panic!("single tensor should be valid"));
    let multi = Tensor4D::from_vec(1, 2, 1, 1, vec![2.0, 5.0])
        .unwrap_or_else(|_| panic!("multi tensor should be valid"));

    let kernel_single = Tensor4D::from_vec(1, 1, 1, 1, vec![3.0])
        .unwrap_or_else(|_| panic!("single kernel should be valid"));
    let kernel_multi = Tensor4D::from_vec(1, 2, 1, 1, vec![3.0, 1.0])
        .unwrap_or_else(|_| panic!("multi kernel should be valid"));

    let out_single = single
        .conv2d_valid(&kernel_single, None, 1, 1)
        .unwrap_or_else(|_| panic!("single conv should succeed"));
    let out_multi = multi
        .conv2d_valid(&kernel_multi, None, 1, 1)
        .unwrap_or_else(|_| panic!("multi conv should succeed"));

    assert_eq!(out_single.shape(), (1, 1, 1, 1));
    assert_eq!(out_multi.shape(), (1, 1, 1, 1));
    assert_eq!(out_single.get(0, 0, 0, 0), Ok(6.0));
    assert_eq!(out_multi.get(0, 0, 0, 0), Ok(11.0));
}

#[test]
fn rag_pipeline_answer_text_respects_top_k() {
    let retriever = StaticRetriever {
        chunks: vec![
            RagChunk {
                document_id: "doc_1".to_string(),
                chunk_id: "doc_1_0".to_string(),
                text: "alpha".to_string(),
                score: 0.9,
            },
            RagChunk {
                document_id: "doc_2".to_string(),
                chunk_id: "doc_2_0".to_string(),
                text: "beta".to_string(),
                score: 0.8,
            },
        ],
    };
    let pipeline = RagPipeline::new(retriever, EchoGenerator, 1);

    let answer = pipeline.answer_text("what is alpha");
    assert_eq!(answer.context.chunks.len(), 1);
    assert_eq!(answer.model_label_hint, None);
    assert_eq!(answer.model_score_hint, None);
}
