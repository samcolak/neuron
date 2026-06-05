#![allow(dead_code)]
#![allow(clippy::wildcard_dependencies, reason = "")]
#![allow(clippy::multiple_crate_versions, reason = "")]
#![allow(clippy::too_many_arguments, reason = "")]
#![allow(clippy::if_same_then_else, reason = "")]
#![allow(clippy::type_complexity, reason = "")]
#![allow(clippy::useless_format, reason = "")]
#![allow(clippy::absolute_paths, reason = "")]
#![allow(clippy::clone_on_ref_ptr, reason = "")]

mod helpers;

use helpers::brain::MultiModalNeuralNetwork;
use helpers::controllers::multimodal_controller::MultiModalInput;

use crate::helpers::text_dendrite::DendriteType;

fn run_multimodal_demo() {

    let mut network = MultiModalNeuralNetwork::new_multimodal();

    network.insert_text("cat on mat", "en", DendriteType::Statement);
    network.insert_text("dog in park", "en", DendriteType::Statement);
    network.absorb_true_text("water is wet", "en", DendriteType::Statement);

    let flat_image: Vec<u8> = vec![16; 256];
    let high_edge_image: Vec<u8> = (0..256)
        .map(|i| if i % 2 == 0 { 0 } else { 255 })
        .collect();

    network.insert_image_bytes(&flat_image, "img", DendriteType::Token);
    network.insert_image_bytes(&high_edge_image, "img", DendriteType::Token);

    let text_hit = network.enumerate_text_path("cat on mat").0;
    let truth_hit = network.enumerate_text_path("water is wet").0;
    let image_hit = network.enumerate_image_bytes_path(&flat_image).0;

    let audio_features = MultiModalInput::FeatureTokens {
        modality: "audio".to_string(),
        tokens: vec!["mfcc0:12".to_string(), "mfcc1:3a".to_string()],
    };
    network.insert_multimodal(&audio_features, "audio", DendriteType::Token);
    let audio_hit = network.enumerate_multimodal_path(&audio_features).0;

    println!("Multimodal demo: {} total nodes", network.all_dendrites_sorted().len());
    println!(
        "  cognitive_nodes={} memory_nodes={}",
        network.cognitive_network().all_dendrites_sorted().len(),
        network.memory_network().all_dendrites_sorted().len(),
    );
    println!(
        "  query[text='cat on mat'] -> {}",
        text_hit
            .as_ref()
            .map(|node| node.data.as_str())
            .unwrap_or("<no hit>")
    );
    println!(
        "  query[truth='water is wet'] -> {}",
        truth_hit
            .as_ref()
            .map(|node| node.data.as_str())
            .unwrap_or("<no hit>")
    );
    println!(
        "  query[image=flat_256] -> {}",
        image_hit
            .as_ref()
            .map(|node| node.data.as_str())
            .unwrap_or("<no hit>")
    );
    println!(
        "  query[audio=mfcc] -> {}",
        audio_hit
            .as_ref()
            .map(|node| node.data.as_str())
            .unwrap_or("<no hit>")
    );

}

fn main() {
    println!("Multimodal brain demo");
    run_multimodal_demo();
}
