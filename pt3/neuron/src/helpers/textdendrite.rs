
use crate::helpers::axon::Axon;
use crate::helpers::nodenet::NetworkNode;

use uuid::Uuid;

use std::collections::HashMap;
use serde::{Serialize, Deserialize};


#[derive(Debug, Copy, Clone, Serialize, Deserialize, Default)]
#[repr(u8)]
pub enum DendriteType {
    #[default]
    Unknown = 0,
    Question = 1,
    Statement = 2,
    Token = 253,
    StopWord = 254,
    Other = 255,
}

impl std::fmt::Display for DendriteType {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let type_str = match self {
            DendriteType::Unknown => "Unknown",
            DendriteType::Question => "Question",
            DendriteType::Statement => "Statement",
            DendriteType::Token => "Token",
            DendriteType::StopWord => "StopWord",
            DendriteType::Other => "Other",
        };
        write!(f, "{}", type_str)
    }
    
}


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
pub struct TextDendrite {
    pub uid: String,
    pub connections: Vec<Axon>,
    pub data: String,
    pub result: Option<String>,
    pub lang: String,
    pub dendrite_type: DendriteType,
    #[serde(skip, default)]
    pub normalized_key: String,
    #[serde(skip, default)]
    pub connection_index: HashMap<String, usize>,
}


impl TextDendrite {

    pub fn unique_id() -> String {    
        Uuid::now_v7().to_string().replace("-", "")    
    }

    pub fn new(data: &str, lang: &str, dendrite_type: DendriteType) -> Self {
        
        let uid = Self::unique_id();
        let normalized_key = normalize_data_key(&data);
        
        Self {
            uid,
            connections: Vec::new(),
            data: data.to_string(),
            lang: lang.to_string(),
            dendrite_type,
            normalized_key,
            result: None,
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

impl NetworkNode for TextDendrite {

    fn new_node(data: &str, language: &str, dendrite_type: DendriteType) -> Self {
        Self::new(data, language, dendrite_type)
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
