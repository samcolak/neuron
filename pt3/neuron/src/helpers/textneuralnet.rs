use crate::helpers::controllers::textnode_controller::TextNodeController;
use crate::helpers::textdendrite::{DendriteType, TextDendrite};
use crate::helpers::neuralnet::NeuralNetwork;
use crate::helpers::nodenet::NodeNetwork;

use lazy_static::lazy_static;

use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

pub type TextNeuralNetwork = NeuralNetwork<TextNodeController, TextDendrite>;

lazy_static! {
    static ref NEURALNET: RwLock<TextNeuralNetwork> = RwLock::new(NeuralNetwork::new());
}

// singleton accessors and mutators for the neural network instance

pub fn get_neural_network_read() -> RwLockReadGuard<'static, TextNeuralNetwork> {
    NEURALNET.read().unwrap()
}

pub fn get_neural_network_write() -> RwLockWriteGuard<'static, TextNeuralNetwork> {
    NEURALNET.write().unwrap()
}

pub fn get_neural_network() -> RwLockWriteGuard<'static, TextNeuralNetwork> {
    get_neural_network_write()
}

pub fn neuralnet_add_dendrite(data: &str, language: &str, dendrite_type: DendriteType) {
    let mut neural_net = get_neural_network_write();
    let _ = neural_net.insert_dendrite_and_index(data, language, dendrite_type);
}

pub fn neuralnet_enumerate_dendrites() -> Vec<TextDendrite> {
    let neural_net = get_neural_network_read();
    neural_net.all_dendrites_sorted()
}

pub fn neuralnet_enumerate(data: &str) -> Vec<TextDendrite> {
    let neural_net = get_neural_network_read();
    neural_net.enumerate_children(data)
}

pub fn neuralnet_insert(content: &str, language: &str, dendrite_type: DendriteType) {
    let mut neural_net = get_neural_network_write();
    neural_net.insert(content, language, dendrite_type);
}

pub fn neuralnet_save(filename: &str) {
    let neural_net = get_neural_network_read();
    neural_net.save(filename);
}

pub fn neuralnet_load(filename: &str) {
    let mut neural_net = get_neural_network_write();
    let _ = neural_net.load(filename);
}

impl NeuralNetwork<TextNodeController, TextDendrite> {
    pub fn new() -> Self {
        Self::with_controller(TextNodeController)
    }

    pub fn enumerate_path(&self, data: &str) -> (Option<TextDendrite>, Vec<TextDendrite>) {
        self.enumerate_path_content(data)
    }

    pub fn insert(&mut self, content: &str, language: &str, dendrite_type: DendriteType) {
        self.insert_content(content, language, dendrite_type)
    }
}
