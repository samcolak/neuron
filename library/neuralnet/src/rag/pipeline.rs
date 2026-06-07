use crate::core::brain::MultiModalBrain;
use crate::helpers::multimodal_controller::MultiModalInput;

use super::bridge::BrainRagBridge;
use super::types::{RagAnswer, RagContext, RagChunk};

pub trait Retriever: Send + Sync {
    fn retrieve(&self, query: &str, top_k: usize) -> Vec<RagChunk>;
}

pub trait Generator: Send + Sync {
    fn generate(&self, question: &str, context: &RagContext) -> String;
}

pub struct RagPipeline<R: Retriever, G: Generator> {
    retriever: R,
    generator: G,
    top_k: usize,
}

impl<R: Retriever, G: Generator> RagPipeline<R, G> {

    pub fn new(retriever: R, generator: G, top_k: usize) -> Self {
        Self {
            retriever,
            generator,
            top_k: top_k.max(1),
        }
    }

    pub fn answer_text(&self, question: &str) -> RagAnswer {

        let query = question.trim();
        let chunks = self.retriever.retrieve(query, self.top_k);
        let context = RagContext {
            query: query.to_string(),
            chunks,
        };
        let answer = self.generator.generate(query, &context);

        RagAnswer {
            answer,
            context,
            model_label_hint: None,
            model_score_hint: None,
        }

    }

    pub fn answer_with_brain(
        &self,
        brain: &MultiModalBrain,
        input: &MultiModalInput,
    ) -> RagAnswer {

        let query = BrainRagBridge::query_from_input(input);
        let chunks = self.retriever.retrieve(query.as_str(), self.top_k);

        let context = RagContext {
            query,
            chunks,
        };

        let answer = self.generator.generate(context.query.as_str(), &context);
        let hint = BrainRagBridge::classify_hint(brain, input);

        RagAnswer {
            answer,
            context,
            model_label_hint: hint.as_ref().map(|c| c.label.clone()),
            model_score_hint: hint.map(|c| c.score),
        }
        
    }
}
