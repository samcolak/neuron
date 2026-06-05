
use crate::helpers::axon::Axon;
use crate::core::nodenet::{NetworkNode, NodeMetadata};
use crate::dendrites::text_dendrite::DendriteType;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageDendrite {
    pub uid: String,
    pub connections: Vec<Axon>,
    pub data: String,
    pub feature_type: String,
    pub metadata: NodeMetadata,
    pub dendrite_type: DendriteType,
    #[serde(skip, default)]
    pub normalized_key: String,
    #[serde(skip, default)]
    pub connection_index: HashMap<String, usize>,
}

impl ImageDendrite {

    pub fn new(data: &str, metadata: &NodeMetadata, dendrite_type: DendriteType) -> Self {
        let uid = Self::unique_id();
        let normalized_key = data.trim().to_ascii_lowercase();

        Self {
            uid,
            connections: Vec::new(),
            data: data.to_string(),
            feature_type: "image_feature".to_string(),
            metadata: metadata.clone(),
            dendrite_type,
            normalized_key,
            connection_index: HashMap::new(),
        }
    }

    pub fn connect(&mut self, other: String, weight: i64) {
        if let Some(existing_index) = self.connection_index.get(&other).copied()
            && let Some(existing_connection) = self.connections.get_mut(existing_index)
        {
            existing_connection.weight += weight;
            return;
        }

        let connection = Axon {
            from: self.uid.clone(),
            to: other.clone(),
            weight,
        };

        self.connections.push(connection);
        let inserted_index = self.connections.len() - 1;
        self.connection_index.insert(other, inserted_index);
    }

}

impl NetworkNode for ImageDendrite {

    fn new_node(data: &str, metadata: &NodeMetadata, dendrite_type: DendriteType) -> Self {
        Self::new(data, metadata, dendrite_type)
    }

    fn uid(&self) -> &str {
        &self.uid
    }

    fn data(&self) -> &str {
        &self.data
    }

    fn normalized_key(&self) -> &str {
        &self.normalized_key
    }

    fn set_normalized_key(&mut self, normalized_key: String) {
        self.normalized_key = normalized_key;
    }

    fn connections(&self) -> &[Axon] {
        &self.connections
    }

    fn connect(&mut self, other: String, weight: i64) {
        self.connect(other, weight);
    }

    fn has_connection_to(&self, to_uid: &str) -> bool {
        self.connection_index.contains_key(to_uid)
    }

    fn rebuild_connection_index(&mut self) {
        self.connection_index.clear();
        for (idx, connection) in self.connections.iter().enumerate() {
            self.connection_index.insert(connection.to.clone(), idx);
        }
    }

}
