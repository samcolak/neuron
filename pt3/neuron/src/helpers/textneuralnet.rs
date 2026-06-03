
use crate::helpers::controllers::textnode_controller::{
    get_first_hit_metrics,
    FirstHitMetrics,
    TextNodeController,
};

use crate::helpers::network_store::{
    build_text_network_store_for_network,
    TextNetworkStore,
    TextNetworkStoreMetrics,
};

use crate::helpers::textdendrite::{DendriteType, TextDendrite};
use crate::helpers::neuralnet::NeuralNetwork;
use crate::helpers::nodenet::NodeNetwork;

use lazy_static::lazy_static;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

pub type TextNeuralNetwork = NeuralNetwork<TextNodeController, TextDendrite>;

pub const DEFAULT_NETWORK_ID: &str = "default";

struct ManagedTextNetwork {
    network: RwLock<TextNeuralNetwork>,
    store: Box<dyn TextNetworkStore>,
}

impl ManagedTextNetwork {
    fn new(network_id: &str) -> Self {
        let store = build_text_network_store_for_network(network_id);
        let mut network = TextNeuralNetwork::new();
        let _ = store.load_into(&mut network);

        Self {
            network: RwLock::new(network),
            store,
        }
    }
}

fn resolve_network_id(network_id: &str) -> String {
    let trimmed = network_id.trim();
    if trimmed.is_empty() {
        DEFAULT_NETWORK_ID.to_string()
    } else {
        trimmed.to_string()
    }
}

fn get_or_create_network(network_id: &str) -> Arc<ManagedTextNetwork> {
    let resolved_id = resolve_network_id(network_id);

    if let Some(existing) = NETWORK_REGISTRY
        .read()
        .unwrap()
        .get(&resolved_id)
        .cloned()
    {
        return existing;
    }

    let mut registry = NETWORK_REGISTRY.write().unwrap();

    if let Some(existing) = registry.get(&resolved_id).cloned() {
        return existing;
    }

    let created = Arc::new(ManagedTextNetwork::new(&resolved_id));
    registry.insert(resolved_id, Arc::clone(&created));
    created
}

lazy_static! {
    static ref DEFAULT_NETWORK: Arc<ManagedTextNetwork> =
        Arc::new(ManagedTextNetwork::new(DEFAULT_NETWORK_ID));
    static ref NETWORK_REGISTRY: RwLock<HashMap<String, Arc<ManagedTextNetwork>>> = {
        let mut map = HashMap::new();
        map.insert(DEFAULT_NETWORK_ID.to_string(), Arc::clone(&DEFAULT_NETWORK));
        RwLock::new(map)
    };
}

// singleton accessors and mutators for the neural network instance

pub fn get_neural_network_read() -> RwLockReadGuard<'static, TextNeuralNetwork> {
    DEFAULT_NETWORK.network.read().unwrap()
}

pub fn get_neural_network_write() -> RwLockWriteGuard<'static, TextNeuralNetwork> {
    DEFAULT_NETWORK.network.write().unwrap()
}

pub fn get_neural_network() -> RwLockWriteGuard<'static, TextNeuralNetwork> {
    get_neural_network_write()
}

pub fn neuralnet_add_dendrite(data: &str, language: &str, dendrite_type: DendriteType) {
    neuralnet_add_dendrite_for(DEFAULT_NETWORK_ID, data, language, dendrite_type);
}

pub fn neuralnet_add_dendrite_for(
    network_id: &str,
    data: &str,
    language: &str,
    dendrite_type: DendriteType,
) {
    let managed = get_or_create_network(network_id);
    let mut neural_net = managed.network.write().unwrap();
    let _ = neural_net.insert_dendrite_and_index(data, language, dendrite_type);
    let snapshot = neural_net.clone();
    drop(neural_net);
    managed.store.persist(&snapshot);
}

pub fn neuralnet_enumerate_dendrites() -> Vec<TextDendrite> {
    neuralnet_enumerate_dendrites_for(DEFAULT_NETWORK_ID)
}

pub fn neuralnet_enumerate_dendrites_for(network_id: &str) -> Vec<TextDendrite> {
    let managed = get_or_create_network(network_id);
    let neural_net = managed.network.read().unwrap();
    neural_net.all_dendrites_sorted()
}

pub fn neuralnet_enumerate(data: &str) -> Vec<TextDendrite> {
    neuralnet_enumerate_for(DEFAULT_NETWORK_ID, data)
}

pub fn neuralnet_enumerate_for(network_id: &str, data: &str) -> Vec<TextDendrite> {
    let managed = get_or_create_network(network_id);
    let neural_net = managed.network.read().unwrap();
    neural_net.enumerate_children(data)
}

pub fn neuralnet_insert(content: &str, language: &str, dendrite_type: DendriteType) {
    neuralnet_insert_for(DEFAULT_NETWORK_ID, content, language, dendrite_type);
}

pub fn neuralnet_insert_for(
    network_id: &str,
    content: &str,
    language: &str,
    dendrite_type: DendriteType,
) {
    let managed = get_or_create_network(network_id);
    let mut neural_net = managed.network.write().unwrap();
    neural_net.insert(content, language, dendrite_type);
    let snapshot = neural_net.clone();
    drop(neural_net);
    managed.store.persist(&snapshot);
}

pub fn neuralnet_save(filename: &str) {
    neuralnet_save_for(DEFAULT_NETWORK_ID, filename);
}

pub fn neuralnet_save_for(network_id: &str, filename: &str) {
    let managed = get_or_create_network(network_id);
    let neural_net = managed.network.read().unwrap();
    neural_net.save(filename);
}

pub fn neuralnet_flush() {
    neuralnet_flush_for(DEFAULT_NETWORK_ID);
}

pub fn neuralnet_flush_for(network_id: &str) {
    let managed = get_or_create_network(network_id);
    let neural_net = managed.network.read().unwrap();
    let snapshot = neural_net.clone();
    drop(neural_net);
    managed.store.persist_force(&snapshot);
}

pub fn neuralnet_store_metrics() -> TextNetworkStoreMetrics {
    neuralnet_store_metrics_for(DEFAULT_NETWORK_ID)
}

pub fn neuralnet_store_metrics_for(network_id: &str) -> TextNetworkStoreMetrics {
    let managed = get_or_create_network(network_id);
    managed.store.metrics()
}

pub fn neuralnet_first_hit_metrics() -> FirstHitMetrics {
    get_first_hit_metrics()
}

pub fn neuralnet_is_loaded(network_id: &str) -> bool {
    let resolved_id = resolve_network_id(network_id);
    NETWORK_REGISTRY
        .read()
        .unwrap()
        .contains_key(&resolved_id)
}

pub fn neuralnet_preload() {
    neuralnet_preload_for(DEFAULT_NETWORK_ID);
}

pub fn neuralnet_preload_for(network_id: &str) {
    let _ = get_or_create_network(network_id);
}

pub fn neuralnet_loaded_network_ids() -> Vec<String> {
    let mut network_ids: Vec<String> = NETWORK_REGISTRY
        .read()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    network_ids.sort();
    network_ids
}

pub fn neuralnet_evict_network(network_id: &str) -> bool {
    
    let resolved_id = resolve_network_id(network_id);

    if resolved_id == DEFAULT_NETWORK_ID {
        return false;
    }

    let removed = NETWORK_REGISTRY.write().unwrap().remove(&resolved_id);

    let Some(managed) = removed else {
        return false;
    };

    let neural_net = managed.network.read().unwrap();
    let snapshot = neural_net.clone();

    drop(neural_net);
    managed.store.persist_force(&snapshot);

    true
}

pub fn neuralnet_load(filename: &str) {
    neuralnet_load_for(DEFAULT_NETWORK_ID, filename);
}

pub fn neuralnet_load_for(network_id: &str, filename: &str) {
    
    let managed = get_or_create_network(network_id);
    let mut neural_net = managed.network.write().unwrap();
    if neural_net.load(filename) {
        let snapshot = neural_net.clone();
        drop(neural_net);
        managed.store.persist_force(&snapshot);
    }

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
