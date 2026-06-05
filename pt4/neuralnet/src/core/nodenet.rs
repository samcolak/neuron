use crate::helpers::axon::Axon;
use crate::dendrites::text_dendrite::DendriteType;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use uuid::Uuid;


#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeMetadata {
    pub attributes: HashMap<String, String>,
}

impl NodeMetadata {

    pub fn new() -> Self {
        Self {
            attributes: HashMap::new(),
        }
    }

    pub fn with_lang(language: &str) -> Self {
        let mut metadata = Self::new();
        metadata.set("lang", language);
        metadata
    }

    pub fn set(&mut self, key: &str, value: &str) {
        self.attributes
            .insert(key.trim().to_ascii_lowercase(), value.to_string());
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.attributes
            .get(&key.trim().to_ascii_lowercase())
            .map(String::as_str)
    }

}

pub trait NetworkNode {

    fn unique_id() -> String {
        Uuid::now_v7().to_string().replace('-', "")
    }

    fn new_node(data: &str, metadata: &NodeMetadata, dendrite_type: DendriteType) -> Self
    where
        Self: Sized;

    fn uid(&self) -> &str;
    fn data(&self) -> &str;
    fn normalized_key(&self) -> &str;
    fn set_normalized_key(&mut self, normalized_key: String);
    fn connections(&self) -> &[Axon];
    fn connect(&mut self, other: String, weight: i64);
    fn has_connection_to(&self, to_uid: &str) -> bool;
    fn rebuild_connection_index(&mut self);

}

pub trait TokenClusterKeyStrategy {
    fn cluster_key_for_token(&self, token_key: &str) -> Option<String>;
}

pub trait NodeNetworkController: Clone + Default + TokenClusterKeyStrategy {
    type Content: ?Sized;

    fn tokenize(&self, content: &Self::Content) -> Vec<String>;
    fn normalize_token(&self, token: &str) -> String;
    fn evaluate_match(&self, left: &str, right: &str) -> (f64, Vec<String>);
    fn stop_words(&self, metadata: &NodeMetadata) -> Vec<&'static str>;
}

pub trait NodeNetwork<C: NodeNetworkController> {
    type Node: NetworkNode;

    fn insert_content(
        &mut self,
        content: &C::Content,
        metadata: &NodeMetadata,
        dendrite_type: DendriteType,
    );
    fn enumerate_path_content(&self, content: &C::Content)
    -> (Option<Self::Node>, Vec<Self::Node>);
}
