use crate::helpers::axon::Axon;
use crate::dendrites::text_dendrite::DendriteType;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};

use uuid::Uuid;


#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeMetadata {
    pub attributes: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NodePayload {
    Text(String),
    ImageBytes(Vec<u8>),
    FeatureTokens {
        modality: String,
        tokens: Vec<String>,
    },
}

impl Default for NodePayload {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl NodePayload {
    pub fn text(value: &str) -> Self {
        Self::Text(value.to_string())
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(value) => Some(value.as_str()),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &str {
        self.as_text().unwrap_or("")
    }
}

impl Display for NodePayload {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text(value) => write!(f, "{}", value),
            Self::ImageBytes(bytes) => write!(f, "<image_bytes:{}>", bytes.len()),
            Self::FeatureTokens { modality, tokens } => {
                write!(f, "<feature_tokens:{}:{}>", modality, tokens.len())
            }
        }
    }
}

impl PartialEq<&str> for NodePayload {
    fn eq(&self, other: &&str) -> bool {
        matches!(self, Self::Text(value) if value == other)
    }
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
