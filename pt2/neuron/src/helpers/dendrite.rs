
use crate::helpers::axon::Axon;

use uuid::Uuid;

use std::collections::HashMap;
use serde::{Serialize, Deserialize};


fn normalize_data_key(input: &str) -> String {

    let lowered = input.to_lowercase();
    let normalized: String = lowered
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric() || ch.is_whitespace() {
                ch
            } else {
                ' '
            }
        })
        .collect();

    normalized
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ")

}

// Dendrite represents the receiving end of a connection in a neural network. It can have multiple connections (Axons) coming into it, and it holds some data.

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dendrite {
    pub uid: String,
    pub connections: Vec<Axon>,
    pub data: String,
    #[serde(skip, default)]
    pub normalized_key: String,
    #[serde(skip, default)]
    pub connection_index: HashMap<String, usize>,
}


impl Dendrite {

    pub fn unique_id() -> String {    
        Uuid::now_v7().to_string().replace("-", "")    
    }

    pub fn new(data: String) -> Self {
        
        let uid = Self::unique_id();
        let normalized_key = normalize_data_key(&data);
        
        Self {
            uid,
            connections: Vec::new(),
            data,
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
