#![allow(dead_code)]
#![allow(clippy::wildcard_dependencies, reason = "")]
#![allow(clippy::multiple_crate_versions, reason = "")]
#![allow(clippy::too_many_arguments, reason = "")]
#![allow(clippy::if_same_then_else, reason = "")]
#![allow(clippy::type_complexity, reason = "")]
#![allow(clippy::useless_format, reason = "")]
#![allow(clippy::absolute_paths, reason = "")]
#![allow(clippy::clone_on_ref_ptr, reason = "")]

mod multimodal_demo;
mod cnn_classifier_walkthrough;
mod trainer_fixtures;
mod trainer_presentation;
mod trainer_walkthrough;
mod rag_walkthrough;
mod rag_dataset_walkthrough;
mod multimodal_tensor_walkthrough;
mod brain_stress_walkthrough;
#[cfg(test)]
mod rag_tensor_tests;
#[cfg(test)]
mod multimodal_tensor_variants_tests;

fn main() {
    println!("Multimodal brain demo");
    println!(
        "Tensor backend in use: {}",
        neuralnet::tensor::backend::active_backend_label()
    );
    multimodal_demo::run_multimodal_demo();
    trainer_walkthrough::run_trainer_walkthrough();
    cnn_classifier_walkthrough::run_cnn_classifier_walkthrough();
    rag_walkthrough::run_rag_walkthrough();
    rag_dataset_walkthrough::run_rag_dataset_walkthrough();
    multimodal_tensor_walkthrough::run_multimodal_tensor_walkthrough();
    brain_stress_walkthrough::run_brain_stress_walkthrough();
}
