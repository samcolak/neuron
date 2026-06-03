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

use std::{collections::HashMap};

use helpers::neuralnet::NeuralNetwork;

use crate::helpers::dendrite::DendriteType;

fn main() {
    
    let mut network = NeuralNetwork::new();

    network.insert("the quick brown fox jumps over the lazy dog", "en", DendriteType::Statement);    
    network.insert("the sun rises over the horizon above the ocean", "en", DendriteType::Statement);
    network.insert("the moon shines brightly in the night sky", "en", DendriteType::Statement);
    network.insert("the stars twinkle in the vast expanse of space", "en", DendriteType::Statement);
    network.insert("the river flows gently through the valley", "en", DendriteType::Statement);
    network.insert("the mountain stands tall against the sky", "en", DendriteType::Statement);
    network.insert("sunlight filters through the leaves of the trees", "en", DendriteType::Statement);
    network.insert("the ocean waves crash against the shore", "en", DendriteType::Statement);
    network.insert("the city lights illuminate the night", "en", DendriteType::Statement);
    network.insert("the flowers bloom in the springtime", "en", DendriteType::Statement);
    network.insert("winter snow blankets the landscape in white", "en", DendriteType::Statement);
    network.insert("lakes reflect the beauty of the surrounding nature", "en", DendriteType::Statement);
    network.insert("the horizon changes into the night", "en", DendriteType::Statement);
    network.insert("my view on the world is shaped by my experiences", "en", DendriteType::Statement);
    network.insert("quick brown dogs are often seen in the park", "en", DendriteType::Statement);
    network.insert("moonlight casts a serene glow over the countryside", "en", DendriteType::Statement);
    network.insert("the stars guide travelers through the night", "en", DendriteType::Statement);
    network.insert("the river's gentle flow soothes the soul", "en", DendriteType::Statement);
    network.insert("the mountain's majestic presence inspires awe", "en", DendriteType::Statement);
    network.insert("sunlight warms the earth and nurtures life", "en", DendriteType::Statement);
    network.insert("the ocean's vastness holds countless mysteries", "en", DendriteType::Statement);
    network.insert("the city buzzes with energy and excitement", "en", DendriteType::Statement);
    network.insert("the flowers' vibrant colors brighten the day", "en", DendriteType::Statement);
    network.insert("the quick brown goat jumps over the lazy cat", "en", DendriteType::Statement);

    network.save("testing.bin");
    // network.load("testing.bin");

    let dendrites = network.all_dendrites_sorted();

    let dendrite_by_uid: HashMap<&str, &str> = dendrites
        .iter()
        .map(|d| (d.uid.as_str(), d.data.as_str()))
        .collect();

    dendrites.iter().for_each(|d| {
        println!("Dendrite: {} | {}", d.data, d.dendrite_type as u8);
        for synapse in &d.connections {
            if let Some(connecting_data) = dendrite_by_uid.get(synapse.to.as_str()) {
                println!("  Synapse: {} (weight: {})", connecting_data, synapse.weight);
            }
        }
    });

    network.enumerate_children("lakes").iter().for_each(|d| {
        println!("Child Dendrite: {}", d.data);
    });

    network.enumerate_path("the stars").1.iter().for_each(|d| {
        println!("Path Dendrite: {}", d.data);
    });

}
