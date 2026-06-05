#![allow(dead_code)]
#![allow(clippy::wildcard_dependencies, reason = "")]
#![allow(clippy::multiple_crate_versions, reason = "")]
#![allow(clippy::too_many_arguments, reason = "")]
#![allow(clippy::if_same_then_else, reason = "")]
#![allow(clippy::type_complexity, reason = "")]
#![allow(clippy::useless_format, reason = "")]
#![allow(clippy::absolute_paths, reason = "")]
#![allow(clippy::clone_on_ref_ptr, reason = "")]

use neuralnet::helpers::brain::MultiModalNeuralNetwork;
use neuralnet::helpers::brain::QuestionStoreDecision;
use neuralnet::helpers::multimodal_controller::MultiModalInput;
use neuralnet::helpers::nodenet::NodeMetadata;
use neuralnet::helpers::text_dendrite::DendriteType;

#[derive(Debug)]
struct ModelCheck {
    label: &'static str,
    actual: bool,
    expected: bool,
}

fn build_multimodal_demo_network() -> MultiModalNeuralNetwork {

    let mut network = MultiModalNeuralNetwork::new_multimodal();
    let text_metadata = NodeMetadata::with_lang("en");
    let image_metadata = NodeMetadata::with_lang("img");
    let audio_metadata = NodeMetadata::with_lang("audio");
    let sensor_metadata = NodeMetadata::with_lang("sensor");
    let vision_metadata = NodeMetadata::with_lang("vision");

    network.insert_text("cat on mat", &text_metadata, DendriteType::Statement);
    network.insert_text("dog in park", &text_metadata, DendriteType::Statement);
    network.insert_text("neuron learns", &text_metadata, DendriteType::Statement);
    network.insert_text("machine reasons", &text_metadata, DendriteType::Statement);
    network.absorb_true_text("water is wet", &text_metadata, DendriteType::Statement);
    network.absorb_true_text("gravity pulls", &text_metadata, DendriteType::Statement);
    network.absorb_true_text("fire is hot", &text_metadata, DendriteType::Statement);

    let flat_image: Vec<u8> = vec![16; 256];
    let truth_image: Vec<u8> = vec![64; 128];
    let signal_image: Vec<u8> = vec![8; 64];
    let high_edge_image: Vec<u8> = (0..256).map(|i| if i % 2 == 0 { 0 } else { 255 }).collect();

    network.insert_image_bytes(&flat_image, &image_metadata, DendriteType::Token);
    network.insert_image_bytes(&high_edge_image, &image_metadata, DendriteType::Token);
    network.insert_image_bytes(&signal_image, &image_metadata, DendriteType::Token);
    network.absorb_true_image_bytes(&truth_image, &image_metadata, DendriteType::Token);

    let audio_features = MultiModalInput::FeatureTokens {
        modality: "audio".to_string(),
        tokens: vec!["mfcc0:12".to_string(), "mfcc1:3a".to_string()],
    };
    network.insert_multimodal(&audio_features, &audio_metadata, DendriteType::Token);

    let sensor_features = MultiModalInput::FeatureTokens {
        modality: "sensor".to_string(),
        tokens: vec!["temp:1f".to_string(), "humidity:08".to_string()],
    };
    network.absorb_truth(&sensor_features, &sensor_metadata, DendriteType::Token);

    let vision_features = MultiModalInput::FeatureTokens {
        modality: "vision".to_string(),
        tokens: vec!["edge:04".to_string(), "shape:09".to_string()],
    };
    network.insert_multimodal(&vision_features, &vision_metadata, DendriteType::Token);

    network

}

fn run_model_checks(network: &MultiModalNeuralNetwork) -> Vec<ModelCheck> {

    let flat_image: Vec<u8> = vec![16; 256];
    let truth_image: Vec<u8> = vec![64; 128];

    let sensor_features = MultiModalInput::FeatureTokens {
        modality: "sensor".to_string(),
        tokens: vec!["temp:1f".to_string(), "humidity:08".to_string()],
    };

    let vision_features = MultiModalInput::FeatureTokens {
        modality: "vision".to_string(),
        tokens: vec!["edge:04".to_string(), "shape:09".to_string()],
    };

    let empty_text = MultiModalInput::Text(String::new());
    let unsupported_features = MultiModalInput::FeatureTokens {
        modality: "unknown".to_string(),
        tokens: Vec::new(),
    };

    let partial_question = MultiModalInput::Text("cat on mat extra".to_string());
    let unknown_question = MultiModalInput::Text("does not exist".to_string());

    let checks = vec![
        ModelCheck {
            label: "cognitive text lookup",
            actual: network.enumerate_text_path("cat on mat").0.is_some(),
            expected: true,
        },
        ModelCheck {
            label: "memory truth lookup",
            actual: network.enumerate_text_path("water is wet").0.is_some(),
            expected: true,
        },
        ModelCheck {
            label: "memory image lookup",
            actual: network.enumerate_image_bytes_path(&truth_image).0.is_some(),
            expected: true,
        },
        ModelCheck {
            label: "feature-token sensor lookup",
            actual: network
                .enumerate_multimodal_path(&sensor_features)
                .0
                .is_some(),
            expected: true,
        },
        ModelCheck {
            label: "feature-token vision lookup",
            actual: network
                .enumerate_multimodal_path(&vision_features)
                .0
                .is_some(),
            expected: true,
        },
        ModelCheck {
            label: "partial question returns a usable score",
            actual: (network.evaluate_question_fuzziness(&partial_question) - 0.75).abs()
                < f64::EPSILON,
            expected: true,
        },
        ModelCheck {
            label: "unknown question is invalid",
            actual: network.evaluate_question_fuzziness(&unknown_question) == -1.0,
            expected: true,
        },
        ModelCheck {
            label: "empty question is invalid",
            actual: network.evaluate_question_fuzziness(&empty_text) == -1.0,
            expected: true,
        },
        ModelCheck {
            label: "unsupported feature payload is invalid",
            actual: network.evaluate_question_fuzziness(&unsupported_features) == -1.0,
            expected: true,
        },
        ModelCheck {
            label: "partial question should be stored",
            actual: network.decide_question_storage(&partial_question)
                == QuestionStoreDecision::Store,
            expected: true,
        },
        ModelCheck {
            label: "unknown question should defer",
            actual: network.decide_question_storage(&unknown_question)
                == QuestionStoreDecision::Defer,
            expected: true,
        },
        ModelCheck {
            label: "known image lookup",
            actual: network.enumerate_image_bytes_path(&flat_image).0.is_some(),
            expected: true,
        },
    ];

    checks

}

fn run_multimodal_demo() {

    let network = build_multimodal_demo_network();

    let text_hit = network.enumerate_text_path("cat on mat").0;
    let truth_hit = network.enumerate_text_path("water is wet").0;
    let machine_hit = network.enumerate_text_path("machine reasons").0;
    let flat_image: Vec<u8> = vec![16; 256];
    let image_hit = network.enumerate_image_bytes_path(&flat_image).0;
    let truth_image: Vec<u8> = vec![64; 128];
    let truth_image_hit = network.enumerate_image_bytes_path(&truth_image).0;
    let audio_features = MultiModalInput::FeatureTokens {
        modality: "audio".to_string(),
        tokens: vec!["mfcc0:12".to_string(), "mfcc1:3a".to_string()],
    };
    let audio_hit = network.enumerate_multimodal_path(&audio_features).0;
    let sensor_features = MultiModalInput::FeatureTokens {
        modality: "sensor".to_string(),
        tokens: vec!["temp:1f".to_string(), "humidity:08".to_string()],
    };
    let sensor_hit = network.enumerate_multimodal_path(&sensor_features).0;
    let vision_features = MultiModalInput::FeatureTokens {
        modality: "vision".to_string(),
        tokens: vec!["edge:04".to_string(), "shape:09".to_string()],
    };
    let vision_hit = network.enumerate_multimodal_path(&vision_features).0;

    println!(
        "Multimodal demo: {} total nodes",
        network.all_dendrites_sorted().len()
    );
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
        "  query[text='machine reasons'] -> {}",
        machine_hit
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
        "  query[truth_image=64x128] -> {}",
        truth_image_hit
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
    println!(
        "  query[sensor=temp] -> {}",
        sensor_hit
            .as_ref()
            .map(|node| node.data.as_str())
            .unwrap_or("<no hit>")
    );
    println!(
        "  query[vision=edge] -> {}",
        vision_hit
            .as_ref()
            .map(|node| node.data.as_str())
            .unwrap_or("<no hit>")
    );

    println!();
    println!("{}", network.format_brain_dump());

    let checks = run_model_checks(&network);
    let passed = checks
        .iter()
        .filter(|check| check.actual == check.expected)
        .count();

    println!("  model checks: {}/{} passed", passed, checks.len());
    for check in checks {
        println!(
            "    [{}] {}",
            if check.actual == check.expected {
                "PASS"
            } else {
                "FAIL"
            },
            check.label
        );
    }

    let snapshot_id = "main_demo_instance";

    match network.snapshot_instance(snapshot_id) {
        Ok(()) => {
            let mut restored = MultiModalNeuralNetwork::new_multimodal();
            match restored.load_snapshot_instance(snapshot_id) {
                Ok(status) => {
                    println!(
                        "  snapshot[{}]: saved and loaded (cognitive_loaded={}, memory_loaded={})",
                        snapshot_id, status.cognitive_loaded, status.memory_loaded
                    );
                    println!(
                        "  snapshot verification: cognitive_nodes={} memory_nodes={}",
                        restored.cognitive_network().all_dendrites_sorted().len(),
                        restored.memory_network().all_dendrites_sorted().len(),
                    );
                }
                Err(err) => {
                    println!("  snapshot[{}]: load failed: {}", snapshot_id, err);
                }
            }
        }
        Err(err) => {
            println!("  snapshot[{}]: save failed: {}", snapshot_id, err);
        }
        
    }

}

fn main() {
    println!("Multimodal brain demo");
    run_multimodal_demo();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_network_keeps_learning_and_truth_separate() {
        let network = build_multimodal_demo_network();

        assert!(
            !network
                .cognitive_network()
                .all_dendrites_sorted()
                .is_empty()
        );
        assert!(!network.memory_network().all_dendrites_sorted().is_empty());
        assert!(network.enumerate_text_path("cat on mat").0.is_some());
        assert!(network.enumerate_text_path("water is wet").0.is_some());
        assert!(network.enumerate_text_path("neuron learns").0.is_some());
        assert!(network.enumerate_text_path("machine reasons").0.is_some());
        assert!(network.enumerate_image_bytes_path(&[64u8; 128]).0.is_some());

        assert!(
            network
                .enumerate_multimodal_path(&MultiModalInput::FeatureTokens {
                    modality: "sensor".to_string(),
                    tokens: vec!["temp:1f".to_string(), "humidity:08".to_string()],
                })
                .0
                .is_some()
        );

        assert!(
            network
                .enumerate_multimodal_path(&MultiModalInput::FeatureTokens {
                    modality: "vision".to_string(),
                    tokens: vec!["edge:04".to_string(), "shape:09".to_string()],
                })
                .0
                .is_some()
        );
    }

    #[test]
    fn demo_network_scores_known_partial_and_unknown_questions() {
        let network = build_multimodal_demo_network();

        let fuzzy = network
            .evaluate_question_fuzziness(&MultiModalInput::Text("cat on mat extra".to_string()));
        let unknown = network
            .evaluate_question_fuzziness(&MultiModalInput::Text("does not exist".to_string()));

        assert_eq!(fuzzy, 0.75);
        assert_eq!(unknown, -1.0);
        assert_eq!(
            network.decide_question_storage(&MultiModalInput::Text("cat on mat extra".to_string())),
            QuestionStoreDecision::Store
        );
        assert_eq!(
            network.decide_question_storage(&MultiModalInput::Text("does not exist".to_string())),
            QuestionStoreDecision::Defer
        );
    }

    #[test]
    fn demo_network_rejects_empty_or_unsupported_inputs() {
        let network = build_multimodal_demo_network();

        let empty_text = network.evaluate_question_fuzziness(&MultiModalInput::Text(String::new()));
        let unsupported = network.evaluate_question_fuzziness(&MultiModalInput::FeatureTokens {
            modality: "unknown".to_string(),
            tokens: Vec::new(),
        });

        assert_eq!(empty_text, -1.0);
        assert_eq!(unsupported, -1.0);
        assert_eq!(
            network.decide_question_storage(&MultiModalInput::Text(String::new())),
            QuestionStoreDecision::Defer
        );
        assert_eq!(
            network.decide_question_storage(&MultiModalInput::FeatureTokens {
                modality: "unknown".to_string(),
                tokens: Vec::new(),
            }),
            QuestionStoreDecision::Defer
        );
    }

    #[test]
    fn demo_network_exposes_a_brain_dump() {
        let network = build_multimodal_demo_network();
        let dump = network.format_brain_dump();

        assert!(dump.contains("Brain dump:"));
        assert!(dump.contains("cognitive network"));
        assert!(dump.contains("memory network"));
        assert!(dump.contains("txt:neuron"));
        assert!(dump.contains("txt:learns"));
        assert!(dump.contains("txt:water"));
        assert!(dump.contains("txt:wet"));
    }
}
