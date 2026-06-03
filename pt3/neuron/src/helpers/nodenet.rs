
use crate::helpers::axon::Axon;
use crate::helpers::text_dendrite::DendriteType;

pub trait NetworkNode {
    fn new_node(data: &str, language: &str, dendrite_type: DendriteType) -> Self
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
    fn stop_words(&self, language: &str) -> Vec<&'static str>;
}

pub trait NodeNetwork<C: NodeNetworkController> {
    type Node: NetworkNode;

    fn insert_content(&mut self, content: &C::Content, language: &str, dendrite_type: DendriteType);
    fn enumerate_path_content(&self, content: &C::Content) -> (Option<Self::Node>, Vec<Self::Node>);
}
