
use crate::helpers::textdendrite::DendriteType;
use crate::helpers::nodenet::{NetworkNode, NodeNetworkController};

use serde::{de::DeserializeOwned, Deserialize, Serialize};

use std::collections::{HashMap, HashSet};
use std::fs;


const NEURON_BIN_MAGIC: [u8; 4] = *b"NRN3";


// code for the core data structures and logic of the neural network

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetwork<C, N>
where
    C: NodeNetworkController,
    N: NetworkNode,
{
    dendrites: HashMap<String, N>,
    #[serde(skip, default)]
    token_index: HashMap<String, Vec<String>>,
    #[serde(skip, default)]
    token_cluster_index: HashMap<String, Vec<String>>,
    #[serde(skip, default)]
    controller: C,
}


enum CandidateUidSet<'a> {
    Borrowed(&'a [String]),
    Owned(Vec<String>),
}


impl<'a> CandidateUidSet<'a> {

    fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    fn as_slice(&self) -> &[String] {
        match self {
            CandidateUidSet::Borrowed(items) => items,
            CandidateUidSet::Owned(items) => items.as_slice(),
        }
    }

}


fn collect_children_from_network<N>(
    dendrites: &HashMap<String, N>,
    data: &str,
) -> Vec<N>
where
    N: NetworkNode + Clone,
{

    let mut children = Vec::new();
    let mut seen_child_uids: HashSet<String> = HashSet::new();

    for parent in dendrites.values().filter(|d| d.data() == data) {
        for connection in parent.connections() {
            if seen_child_uids.insert(connection.to.clone())
                && let Some(child) = dendrites.get(&connection.to)
            {
                children.push(child.clone());
            }
        }
    }

    children
}


impl<C, N> NeuralNetwork<C, N>
where
    C: NodeNetworkController,
    N: NetworkNode + Clone + Serialize + DeserializeOwned,
{

    pub(crate) fn token_key_for_index(&self, data: &str) -> String {
        self.controller.normalize_token(data)
    }

    pub(crate) fn tokenize_content(&self, content: &C::Content) -> Vec<String> {
        self.controller.tokenize(content)
    }

    pub(crate) fn controller(&self) -> &C {
        &self.controller
    }

    fn candidate_uids_for_token<'a>(&'a self, token_key: &str) -> CandidateUidSet<'a> {

        if let Some(exact_matches) = self.token_index.get(token_key) {
            return CandidateUidSet::Borrowed(exact_matches.as_slice());
        }

        let Some(cluster_key) = self.controller.cluster_key_for_token(token_key) else {
            return CandidateUidSet::Owned(Vec::new());
        };

        let Some(cluster_matches) = self.token_cluster_index.get(&cluster_key) else {
            return CandidateUidSet::Owned(Vec::new());
        };

        CandidateUidSet::Owned(cluster_matches
            .iter()
            .filter_map(|uid| {
                let candidate = self.dendrites.get(uid)?;
                let candidate_key_owned;
                let candidate_key = if candidate.normalized_key().is_empty() {
                    candidate_key_owned = self.token_key_for_index(candidate.data());
                    candidate_key_owned.as_str()
                } else {
                    candidate.normalized_key()
                };

                let (score, _) = self.controller.evaluate_match(token_key, candidate_key);

                if score >= 0.60 {
                    Some(uid.clone())
                } else {
                    None
                }
            })
            .collect())

    }

    pub(crate) fn candidate_uids_for_token_vec(&self, token_key: &str) -> Vec<String> {
        self.candidate_uids_for_token(token_key).as_slice().to_vec()
    }

    pub fn load(&mut self, filename: &str) -> bool {

        if let Ok(bytes) = fs::read(filename) {

            if bytes.len() >= 4 && bytes[0..4] == NEURON_BIN_MAGIC
                && let Ok(loaded) = bincode::deserialize::<NeuralNetwork<C, N>>(&bytes[4..])
            {
                *self = loaded;
                self.rebuild_connection_indexes();
                self.rebuild_token_index();
                return true;
            }

            if let Ok(loaded) = serde_json::from_slice::<NeuralNetwork<C, N>>(&bytes) {
                *self = loaded;
                self.rebuild_connection_indexes();
                self.rebuild_token_index();
                return true;
            }

        }

        false
        
    }

    pub fn all_dendrites_sorted(&self) -> Vec<N> {
        
        let mut dendrites: Vec<N> = self.dendrites.values().cloned().collect();
        dendrites.sort_by(|a, b| a.uid().cmp(b.uid()));
        dendrites

    }

    pub fn enumerate_children(&self, data: &str) -> Vec<N> {
        collect_children_from_network(&self.dendrites, data)
    }

    pub fn save(&self, filename: &str) {

        if let Ok(encoded) = bincode::serialize(self) {
            let mut bytes = Vec::with_capacity(NEURON_BIN_MAGIC.len() + encoded.len());
            bytes.extend_from_slice(&NEURON_BIN_MAGIC);
            bytes.extend_from_slice(&encoded);
            let _ = fs::write(filename, bytes);
        }

    }

    pub fn with_controller(controller: C) -> Self {
        Self {
            dendrites: HashMap::new(),
            token_index: HashMap::new(),
            token_cluster_index: HashMap::new(),
            controller,
        }
    }

    pub(crate) fn index_dendrite_token(&mut self, uid: &str) {

        if let Some(dendrite) = self.dendrites.get_mut(uid) {

            if dendrite.normalized_key().is_empty() {
                dendrite.set_normalized_key(self.controller.normalize_token(dendrite.data()));
            }

            let key = dendrite.normalized_key().to_string();

            if key.is_empty() {
                return;
            }

            self.token_index
                .entry(key.clone())
                .or_default()
                .push(uid.to_string());

            if let Some(cluster_key) = self.controller.cluster_key_for_token(&key) {
                self.token_cluster_index
                    .entry(cluster_key)
                    .or_default()
                    .push(uid.to_string());
            }
            
        }

    }

    pub(crate) fn rebuild_token_index(&mut self) {

        self.token_index.clear();
        self.token_cluster_index.clear();
        
        let controller = self.controller.clone();

        for (uid, dendrite) in &mut self.dendrites {

            if dendrite.normalized_key().is_empty() {
                dendrite.set_normalized_key(controller.normalize_token(dendrite.data()));
            }

            let key = dendrite.normalized_key().to_string();

            if key.is_empty() {
                continue;
            }

            self.token_index.entry(key.clone()).or_default().push(uid.clone());

            if let Some(cluster_key) = controller.cluster_key_for_token(&key) {
                self.token_cluster_index
                    .entry(cluster_key)
                    .or_default()
                    .push(uid.clone());
            }
        }

    }

    pub(crate) fn rebuild_connection_indexes(&mut self) {

        for dendrite in self.dendrites.values_mut() {
            dendrite.rebuild_connection_index();
        }

    }

    pub(crate) fn ensure_token_index(&mut self) {
        if self.token_index.is_empty() && !self.dendrites.is_empty() {
            self.rebuild_token_index();
        }
    }

    pub(crate) fn has_direct_connection(&self, from_uid: &str, to_uid: &str) -> bool {
        self.dendrites
            .get(from_uid)
            .map(|from| from.has_connection_to(to_uid))
            .unwrap_or(false)
    }

    pub(crate) fn direct_connection_weight(&self, from_uid: &str, to_uid: &str) -> Option<i64> {
        let from_node = self.dendrites.get(from_uid)?;
        from_node
            .connections()
            .iter()
            .find(|conn| conn.to == to_uid)
            .map(|conn| conn.weight)
    }

    pub(crate) fn best_connected_candidate_uid(
        &self,
        from_uid: &str,
        candidate_uids: &[String],
        min_weight: i64,
    ) -> Option<String> {
        let from_node = self.dendrites.get(from_uid)?;
        let candidate_set: HashSet<&str> = candidate_uids.iter().map(String::as_str).collect();

        from_node
            .connections()
            .iter()
            .filter(|conn| conn.weight >= min_weight && candidate_set.contains(conn.to.as_str()))
            .max_by_key(|conn| conn.weight)
            .map(|conn| conn.to.clone())
    }

    pub(crate) fn connected_uid_by_key(&self, from_uid: &str, target_key: &str) -> Option<String> {
        let from_node = self.dendrites.get(from_uid)?;
        let target_uids = self.candidate_uids_for_token_vec(target_key);

        if target_uids.is_empty() {
            return None;
        }

        from_node.connections().iter().find_map(|conn| {
            if target_uids.iter().any(|uid| uid == &conn.to) {
                Some(conn.to.clone())
            } else {
                None
            }
        })
    }

    pub(crate) fn insert_dendrite_and_index(&mut self, data: &str, language: &str, dendrite_type: DendriteType) -> String {
        let new_dendrite = N::new_node(data, language, dendrite_type);
        let new_uid = new_dendrite.uid().to_string();
        self.dendrites.insert(new_uid.clone(), new_dendrite);
        self.index_dendrite_token(&new_uid);
        new_uid
    }

    pub(crate) fn connect_dendrites(&mut self, from_uid: &str, to_uid: &str, weight: i64) {
        if let Some(from_node) = self.dendrites.get_mut(from_uid) {
            from_node.connect(to_uid.to_string(), weight);
        }
    }

    pub(crate) fn dendrites(&self) -> &HashMap<String, N> {
        &self.dendrites
    }

    pub(crate) fn dendrites_mut(&mut self) -> &mut HashMap<String, N> {
        &mut self.dendrites
    }

}


