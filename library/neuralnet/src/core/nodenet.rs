use crate::helpers::axon::Axon;
use crate::dendrites::text_dendrite::DendriteType;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Display, Formatter};

use uuid::Uuid;


#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContentProvenance {
    pub source_id: Option<String>,
    pub source_uri: Option<String>,
    pub owner_controller: Option<String>,
    pub license: Option<String>,
    pub usage_rights: Option<String>,
    pub lawful_basis: Option<String>,
    pub purpose: Option<String>,
    pub retention_policy: Option<String>,
    pub collected_at_utc: Option<String>,
    pub collector: Option<String>,
    pub jurisdiction: Option<String>,
    pub content_hash: Option<String>,
    pub transformation_lineage: Vec<String>,
}

impl ContentProvenance {
    pub fn has_minimum_attribution(&self) -> bool {
        (self.source_id.is_some() || self.source_uri.is_some())
            && self.owner_controller.is_some()
            && (self.license.is_some() || self.usage_rights.is_some())
    }
}


#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeMetadata {
    #[serde(default)]
    pub attributes: HashMap<String, String>,
    #[serde(default)]
    pub provenance: Option<ContentProvenance>,
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
            provenance: None,
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

    pub fn set_provenance(&mut self, provenance: ContentProvenance) {
        self.provenance = Some(provenance);
    }

    pub fn provenance(&self) -> Option<&ContentProvenance> {
        self.provenance.as_ref()
    }

    pub fn has_minimum_provenance_for_review(&self) -> bool {
        self.provenance
            .as_ref()
            .map(ContentProvenance::has_minimum_attribution)
            .unwrap_or(false)
    }

    pub fn with_source_owner(
        mut self,
        source_id: Option<&str>,
        source_uri: Option<&str>,
        owner_controller: &str,
    ) -> Self {
        let mut provenance = self.provenance.take().unwrap_or_default();
        provenance.source_id = source_id.map(ToString::to_string);
        provenance.source_uri = source_uri.map(ToString::to_string);
        provenance.owner_controller = Some(owner_controller.to_string());
        self.provenance = Some(provenance);
        self
    }

    pub fn with_usage_terms(
        mut self,
        license: Option<&str>,
        usage_rights: Option<&str>,
        lawful_basis: Option<&str>,
    ) -> Self {
        let mut provenance = self.provenance.take().unwrap_or_default();
        provenance.license = license.map(ToString::to_string);
        provenance.usage_rights = usage_rights.map(ToString::to_string);
        provenance.lawful_basis = lawful_basis.map(ToString::to_string);
        self.provenance = Some(provenance);
        self
    }

    pub fn push_transformation_step(&mut self, step: &str) {
        let mut provenance = self.provenance.take().unwrap_or_default();
        provenance.transformation_lineage.push(step.to_string());
        self.provenance = Some(provenance);
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_review_check_requires_source_owner_and_usage_terms() {
        let metadata = NodeMetadata::with_lang("en")
            .with_source_owner(Some("doc-123"), Some("https://example.test/doc"), "example-owner")
            .with_usage_terms(Some("CC-BY-4.0"), None, None);

        assert!(metadata.has_minimum_provenance_for_review());
    }

    #[test]
    fn metadata_review_check_fails_without_owner() {
        let mut metadata = NodeMetadata::with_lang("en");
        metadata.set_provenance(ContentProvenance {
            source_id: Some("doc-123".to_string()),
            source_uri: None,
            owner_controller: None,
            license: Some("CC-BY-4.0".to_string()),
            usage_rights: None,
            lawful_basis: None,
            purpose: None,
            retention_policy: None,
            collected_at_utc: None,
            collector: None,
            jurisdiction: None,
            content_hash: None,
            transformation_lineage: Vec::new(),
        });

        assert!(!metadata.has_minimum_provenance_for_review());
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
