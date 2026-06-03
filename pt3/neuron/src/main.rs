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

use std::{collections::HashMap, env, time::Instant};

use helpers::textdendrite::TextDendrite;
use helpers::controllers::ngram_controller::NgramController;
use helpers::neuralnet::NeuralNetwork;
use helpers::nodenet::NodeNetwork;

use crate::helpers::textdendrite::DendriteType;

fn sample_corpus() -> Vec<&'static str> {

    vec![
        "the quick brown fox jumps over the lazy dog",
        "the sun rises over the horizon above the ocean",
        "the moon shines brightly in the night sky",
        "the stars twinkle in the vast expanse of space",
        "the river flows gently through the valley",
        "the mountain stands tall against the sky",
        "sunlight filters through the leaves of the trees",
        "the ocean waves crash against the shore",
        "the city lights illuminate the night",
        "the flowers bloom in the springtime",
        "winter snow blankets the landscape in white",
        "lakes reflect the beauty of the surrounding nature",
        "the horizon changes into the night",
        "my view on the world is shaped by my experiences",
        "quick brown dogs are often seen in the park",
        "moonlight casts a serene glow over the countryside",
        "the stars guide travelers through the night",
        "the river's gentle flow soothes the soul",
        "the mountain's majestic presence inspires awe",
        "sunlight warms the earth and nurtures life",
        "the ocean's vastness holds countless mysteries",
        "the city buzzes with energy and excitement",
        "the flowers' vibrant colors brighten the day",
        "the quick brown goat jumps over the lazy cat",
    ]

}

fn speed_enabled() -> bool {
    match env::var("NEURON_ENABLE_SPEED") {
        Ok(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on"
        }
        Err(_) => false,
    }
}

fn speed_iterations() -> usize {
    env::var("NEURON_SPEED_ITERATIONS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(200)
}

fn run_text_demo(corpus: &[&str], enable_speed: bool, iterations: usize) {

    let mut network = NeuralNetwork::new();

    for phrase in corpus {
        network.insert(phrase, "en", DendriteType::Statement);
    }

    let dendrites = network.all_dendrites_sorted();
    let dendrite_by_uid: HashMap<&str, &str> = dendrites
        .iter()
        .map(|d| (d.uid.as_str(), d.data.as_str()))
        .collect();

    println!("Text controller: {} nodes", dendrites.len());

    for d in dendrites.iter().take(5) {
        println!("  Dendrite: {} | {}", d.data, d.dendrite_type as u8);
        for synapse in &d.connections {
            if let Some(connecting_data) = dendrite_by_uid.get(synapse.to.as_str()) {
                println!("    Synapse: {} (weight: {})", connecting_data, synapse.weight);
            }
        }
    }

    if enable_speed {

        let mut speed_network = NeuralNetwork::new();
        let insert_start = Instant::now();
        for _ in 0..iterations {
            for phrase in corpus {
                speed_network.insert(phrase, "en", DendriteType::Statement);
            }
        }
        let insert_elapsed = insert_start.elapsed();

        let query_start = Instant::now();
        for _ in 0..iterations {
            let _ = speed_network.enumerate_path("the stars");
        }
        let query_elapsed = query_start.elapsed();

        println!(
            "Text speed: insert={}ms query={}ms (iterations={})",
            insert_elapsed.as_millis(),
            query_elapsed.as_millis(),
            iterations
        );
    
    }

}

fn run_ngram_demo(corpus: &[&str], enable_speed: bool, iterations: usize) {

    let mut network: NeuralNetwork<NgramController, TextDendrite> =
        NeuralNetwork::with_controller(NgramController);

    for phrase in corpus {
        network.insert_content(phrase, "en", DendriteType::Token);
    }

    let nodes = network.all_dendrites_sorted();
    
    println!("Ngram controller: {} nodes", nodes.len());

    for node in nodes.iter().take(8) {
        println!("  Ngram node: {}", node.data);
    }

    if enable_speed {

        let mut speed_network: NeuralNetwork<NgramController, TextDendrite> =
            NeuralNetwork::with_controller(NgramController);

        let insert_start = Instant::now();
        for _ in 0..iterations {
            for phrase in corpus {
                speed_network.insert_content(phrase, "en", DendriteType::Token);
            }
        }
        let insert_elapsed = insert_start.elapsed();

        let query_start = Instant::now();
        for _ in 0..iterations {
            let _ = speed_network.enumerate_path_content("neuralnetwork");
        }
        let query_elapsed = query_start.elapsed();

        println!(
            "Ngram speed: insert={}ms query={}ms (iterations={})",
            insert_elapsed.as_millis(),
            query_elapsed.as_millis(),
            iterations
        );

    }

}

fn main() {
    
    let corpus = sample_corpus();
    let enable_speed = speed_enabled();
    let iterations = speed_iterations();

    println!("Controller comparison demo");
    println!("Speed checks enabled: {}", enable_speed);

    run_text_demo(&corpus, enable_speed, iterations);
    run_ngram_demo(&corpus, enable_speed, iterations);

}
