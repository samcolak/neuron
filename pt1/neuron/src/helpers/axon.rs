

use lazy_static::lazy_static;

use serde::{Deserialize, Serialize};

use std::collections::{HashMap, HashSet};
use std::fs;
use std::sync::Mutex;

const NEURON_BIN_MAGIC: [u8; 4] = *b"NRN1";


lazy_static! {
    static ref NEURALNET: Mutex<NeuralNetwork> = Mutex::new(NeuralNetwork::new());
}


// singleton accessors and mutators for the neural network instance

pub fn get_neural_network() -> std::sync::MutexGuard<'static, NeuralNetwork> {
    NEURALNET.lock().unwrap()
}


pub fn neuralnet_add_dendrite(uid: String, data: String) {
    let mut neural_net = get_neural_network();
    let dendrite = Dendrite::new(uid.clone(), data);
    neural_net.dendrites.insert(uid.clone(), dendrite);
    neural_net.index_dendrite_token(&uid);
}


pub fn neuralnet_enumerate_dendrites() -> Vec<Dendrite> {
    let neural_net = get_neural_network();
    neural_net.all_dendrites_sorted()
}


pub fn neuralnet_enumerate(data: &str) -> Vec<Dendrite> {
    let neural_net = get_neural_network();
    collect_children_from_network(&neural_net.dendrites, data)
}


pub fn neuralnet_insert(content: &str, language: &str) {
    let mut neural_net = get_neural_network();
    neural_net.insert(content, language);
}


pub fn neuralnet_save(filename: &str) {
    let neural_net = get_neural_network();
    neural_net.save(filename);
}


pub fn neuralnet_load(filename: &str) {
    let mut neural_net = get_neural_network();
    let _ = neural_net.load(filename);
}




// code for the core data structures and logic of the neural network

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetwork {
    dendrites: HashMap<String, Dendrite>,
    #[serde(skip, default)]
    token_index: HashMap<String, Vec<String>>,
}


fn collect_children_from_network(
    dendrites: &HashMap<String, Dendrite>,
    data: &str,
) -> Vec<Dendrite> {
    let mut children = Vec::new();
    let mut seen_child_uids: HashSet<String> = HashSet::new();

    for parent in dendrites.values().filter(|d| d.data == data) {
        for connection in &parent.connections {
            if seen_child_uids.insert(connection.to.clone()) {
                if let Some(child) = dendrites.get(&connection.to) {
                    children.push(child.clone());
                }
            }
        }
    }

    children
}


fn evaluate_fuzziness(content: &str, data: &str) -> (f64, Vec<String>) {

    let contentlower = content.to_lowercase();
    let datalower = data.to_lowercase();

    if contentlower == datalower {
        return (1.0, Vec::new());
    }

    if contentlower.contains(&datalower) {
        let mut pieces = Vec::new();
        let index = contentlower.find(&datalower).unwrap();
        if index > 0 {
            pieces.push(content[0..index].to_string());
        }
        let end_index = index + datalower.len();
        if end_index < content.len() {
            pieces.push(content[end_index..].to_string());
        }
        return (0.8, pieces);
    }

    if datalower.contains(&contentlower) {
        let mut pieces = Vec::new();
        let index = datalower.find(&contentlower).unwrap();
        if index > 0 {
            pieces.push(data[0..index].to_string());
        }
        let end_index = index + contentlower.len();
        if end_index < data.len() {
            pieces.push(data[end_index..].to_string());
        }
        return (0.8, pieces);
    }

    (0.0, vec![content.to_string()])
}


fn compare_elememts_for_match(content: &str, data: Vec<Dendrite>) -> (Option<Dendrite>, Vec<String>) {

    let cleaned_content = content.trim();
    let mut resultset = Vec::new();

    for dataitem in data {
        let (fuzziness_score, details) = evaluate_fuzziness(cleaned_content, &dataitem.data);

        if fuzziness_score == 1.0 {
            return (Some(dataitem), details);
        }

        resultset.push((fuzziness_score, dataitem, details));
    }

    if resultset.is_empty() {
        return (None, vec![cleaned_content.to_string()]);
    }

    resultset.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let (best_score, best_match, best_details) = &resultset[0];

    if *best_score > 0.0 {
        return (Some(best_match.clone()), best_details.clone());
    }

    (None, vec![cleaned_content.to_string()])
    
}


fn stop_words_for_language(language: &str) -> Vec<&'static str> {
    match language {
        "en" => vec![
            "a", "an", "and", "as", "at", "by", "for", "from", "in", "is", "of", "on",
            "or", "that", "the", "this", "to", "with",
        ],
        _ => Vec::new(),
    }
}


impl NeuralNetwork {

    pub fn load(&mut self, filename: &str) -> bool {
        if let Ok(bytes) = fs::read(filename) {
            if bytes.len() >= 4 && bytes[0..4] == NEURON_BIN_MAGIC {
                if let Ok(loaded) = bincode::deserialize::<NeuralNetwork>(&bytes[4..]) {
                    *self = loaded;
                    self.rebuild_connection_indexes();
                    self.rebuild_token_index();
                    return true;
                }
            }

            if let Ok(loaded) = serde_json::from_slice::<NeuralNetwork>(&bytes) {
                *self = loaded;
                self.rebuild_connection_indexes();
                self.rebuild_token_index();
                return true;
            }
        }
        false
    }

    pub fn all_dendrites_sorted(&self) -> Vec<Dendrite> {
        let mut dendrites: Vec<Dendrite> = self.dendrites.values().cloned().collect();
        dendrites.sort_by(|a, b| a.uid.cmp(&b.uid));
        dendrites
    }

    pub fn enumerate_children(&self, data: &str) -> Vec<Dendrite> {
        collect_children_from_network(&self.dendrites, data)
    }

    pub fn enumerate_path(&self, data: &str) -> (Option<Dendrite>, Vec<Dendrite>) {
        let path_tokens: Vec<String> = data
            .split_whitespace()
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .map(|token| token.to_lowercase())
            .collect();

        if path_tokens.is_empty() {
            return (None, Vec::new());
        }

        let mut current_uids = self
            .token_index
            .get(&path_tokens[0])
            .cloned()
            .unwrap_or_default();

        for segment_key in &path_tokens[1..] {
            let Some(target_uids) = self.token_index.get(segment_key) else {
                return (None, Vec::new());
            };
            let target_uid_set: HashSet<&str> = target_uids.iter().map(String::as_str).collect();

            let mut next_uids = Vec::new();
            for uid in &current_uids {
                let Some(dendrite) = self.dendrites.get(uid) else {
                    continue;
                };

                for connection in &dendrite.connections {
                    if target_uid_set.contains(connection.to.as_str()) {
                        next_uids.push(connection.to.clone());
                    }
                }
            }

            current_uids = next_uids;
            if current_uids.is_empty() {
                return (None, Vec::new());
            }
        }

        if let Some(last_uid) = current_uids.last() {
            if let Some(last) = self.dendrites.get(last_uid) {
                let mut optional_path = Vec::new();
                for connection in &last.connections {
                    if let Some(next) = self.dendrites.get(&connection.to) {
                        optional_path.push(next.clone());
                    }
                }
                return (Some(last.clone()), optional_path);
            }
        }

        (None, Vec::new())

    }

    pub fn save(&self, filename: &str) {
        if let Ok(encoded) = bincode::serialize(self) {
            let mut bytes = Vec::with_capacity(NEURON_BIN_MAGIC.len() + encoded.len());
            bytes.extend_from_slice(&NEURON_BIN_MAGIC);
            bytes.extend_from_slice(&encoded);
            let _ = fs::write(filename, bytes);
        }
    }

    pub fn insert(&mut self, content: &str, language: &str) {

        if self.token_index.is_empty() && !self.dendrites.is_empty() {
            self.rebuild_token_index();
        }

        fn allocate_uid(neural_net: &NeuralNetwork, next_uid_index: &mut usize) -> String {
            let mut candidate = format!("dendrite_{}", *next_uid_index);
            while neural_net.dendrites.contains_key(&candidate) {
                *next_uid_index += 1;
                candidate = format!("dendrite_{}", *next_uid_index);
            }
            *next_uid_index += 1;
            candidate
        }

        fn has_direct_connection(neural_net: &NeuralNetwork, from_uid: &str, to_uid: &str) -> bool {

            neural_net
                .dendrites
                .get(from_uid)
                .map(|from| from.connection_index.contains_key(to_uid))
                .unwrap_or(false)
                
        }

        fn candidate_uids_for_token<'a>(
            neural_net: &'a NeuralNetwork,
            token_key: &str,
        ) -> Option<&'a Vec<String>> {

            neural_net.token_index.get(token_key)

        }

        fn pick_best_uid(
            neural_net: &NeuralNetwork,
            token_key: &str,
            previous_uid: Option<&str>,
            next_token_key: Option<&str>,
        ) -> Option<String> {

            let candidates = candidate_uids_for_token(neural_net, token_key)?;
            let next_candidates = next_token_key.and_then(|t| candidate_uids_for_token(neural_net, t));

            candidates
                .iter()
                .max_by_key(|candidate_uid| {
                    let mut score = 0;
                    if let Some(prev) = previous_uid {
                        if has_direct_connection(neural_net, prev, candidate_uid) {
                            score += 2;
                        }
                    }
                    if let Some(ref next) = next_candidates {
                        if next
                            .iter()
                            .any(|next_uid| has_direct_connection(neural_net, candidate_uid, next_uid))
                        {
                            score += 1;
                        }
                    }
                    score
                })
                .cloned()
        }

        fn find_connected_uid_by_data(
            neural_net: &NeuralNetwork,
            from_uid: &str,
            target_key: &str,
        ) -> Option<String> {
            let from_node = neural_net.dendrites.get(from_uid)?;
            let target_uids = neural_net.token_index.get(target_key)?;

            from_node.connections.iter().find_map(|conn| {
                if target_uids.iter().any(|uid| uid == &conn.to) {
                    Some(conn.to.clone())
                } else {
                    None
                }
            })
        }

        let tokens: Vec<String> = content
            .split_whitespace()
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .map(|token| token.to_string())
            .collect();

        if tokens.is_empty() {
            return;
        }

        let token_keys: Vec<String> = tokens.iter().map(|token| token.to_lowercase()).collect();

        let neural_net = self;

        let stop_word_set: HashSet<&'static str> = stop_words_for_language(language).into_iter().collect();

        let is_stop_word = |token_key: &str| stop_word_set.contains(token_key);

        let mut selected_existing_path = Vec::new();
        let mut previous_uid: Option<String> = None;

        for index in 0..tokens.len() {
            if is_stop_word(&token_keys[index]) {
                selected_existing_path.clear();
                break;
            }

            let next_token = token_keys.get(index + 1).map(String::as_str);
            let selected = pick_best_uid(
                &neural_net,
                &token_keys[index],
                previous_uid.as_deref(),
                next_token,
            );

            match selected {
                Some(uid) => {
                    previous_uid = Some(uid.clone());
                    selected_existing_path.push(uid);
                }
                None => {
                    selected_existing_path.clear();
                    break;
                }
            }
        }

        let has_complete_existing_path = selected_existing_path.len() == tokens.len()
            && selected_existing_path
                .windows(2)
                .all(|pair| has_direct_connection(&neural_net, &pair[0], &pair[1]));

        let mut next_uid_index = neural_net.dendrites.len() + 1;
        let mut chosen_path: Vec<String> = if has_complete_existing_path {
            selected_existing_path
        } else {
            Vec::new()
        };

        for index in 0..tokens.len() {
            if has_complete_existing_path {
                break;
            }

            if is_stop_word(&token_keys[index]) {
                if let Some(previous_uid) = chosen_path.last() {
                    if let Some(existing_stop_uid) =
                        find_connected_uid_by_data(&neural_net, previous_uid, &token_keys[index])
                    {
                        chosen_path.push(existing_stop_uid);
                        continue;
                    }
                }

                let new_uid = allocate_uid(&neural_net, &mut next_uid_index);
                let new_dendrite = Dendrite::new(new_uid.clone(), tokens[index].clone());
                neural_net.dendrites.insert(new_uid.clone(), new_dendrite);
                neural_net.index_dendrite_token(&new_uid);
                chosen_path.push(new_uid);
                continue;
            }

            let next_token = token_keys.get(index + 1).map(String::as_str);
            let selected = pick_best_uid(
                &neural_net,
                &token_keys[index],
                chosen_path.last().map(String::as_str),
                next_token,
            );

            let uid = match selected {
                Some(existing_uid) => existing_uid,
                None => {
                    let new_uid = allocate_uid(&neural_net, &mut next_uid_index);
                    let new_dendrite = Dendrite::new(new_uid.clone(), tokens[index].clone());
                    neural_net.dendrites.insert(new_uid.clone(), new_dendrite);
                    neural_net.index_dendrite_token(&new_uid);
                    new_uid
                }
            };

            chosen_path.push(uid);
            
        }

        for pair in chosen_path.windows(2) {
            if let Some(from_node) = neural_net.dendrites.get_mut(&pair[0]) {
                from_node.connect(pair[1].clone(), 1);
            }
        }

    }

    pub fn new() -> Self {
        Self {
            dendrites: HashMap::new(),
            token_index: HashMap::new(),
        }
    }

    fn index_dendrite_token(&mut self, uid: &str) {
        if let Some(dendrite) = self.dendrites.get(uid) {
            let key = dendrite.data.to_lowercase();
            self.token_index.entry(key).or_default().push(uid.to_string());
        }
    }

    fn rebuild_token_index(&mut self) {
        self.token_index.clear();
        for (uid, dendrite) in &self.dendrites {
            self.token_index
                .entry(dendrite.data.to_lowercase())
                .or_default()
                .push(uid.clone());
        }
    }

    fn rebuild_connection_indexes(&mut self) {
        for dendrite in self.dendrites.values_mut() {
            dendrite.connection_index.clear();
            for (idx, connection) in dendrite.connections.iter().enumerate() {
                dendrite.connection_index.insert(connection.to.clone(), idx);
            }
        }
    }

}


// Axon represents the sending end of a connection in a neural network. It connects one Dendrite to another and has a weight associated with it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Axon {
    pub from: String,
    pub to: String,
    pub weight: i64,
}

impl Axon {
    
    pub fn new(from: String, to: String, weight: i64) -> Self {
        Self { from, to, weight }
    }

}


// Dendrite represents the receiving end of a connection in a neural network. It can have multiple connections (Axons) coming into it, and it holds some data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dendrite {
    pub uid: String,
    pub connections: Vec<Axon>,
    pub data: String,
    #[serde(skip, default)]
    connection_index: HashMap<String, usize>,
}


impl Dendrite {

    pub fn new(uid: String, data: String) -> Self {
        Self {
            uid,
            connections: Vec::new(),
            data,
            connection_index: HashMap::new(),
        }
    }

    pub fn connect(&mut self, other: String, weight: i64) {
        if let Some(existing_index) = self.connection_index.get(&other).copied() {
            if let Some(existing_connection) = self.connections.get_mut(existing_index) {
                existing_connection.weight += weight;
                return;
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn seeded_network(entries: &[(&str, &str)]) -> NeuralNetwork {
        let mut network = NeuralNetwork::new();
        for (uid, data) in entries {
            network
                .dendrites
                .insert((*uid).to_string(), Dendrite::new((*uid).to_string(), (*data).to_string()));
        }
        network
    }

    #[test]
    fn evaluate_fuzziness_returns_perfect_match() {
        let (score, details) = evaluate_fuzziness("Neuron", "neuron");
        assert_eq!(score, 1.0);
        assert!(details.is_empty());
    }

    #[test]
    fn evaluate_fuzziness_extracts_before_and_after_when_data_contains_content() {
        let (score, details) = evaluate_fuzziness("quick", "the quick fox");
        assert_eq!(score, 0.8);
        assert_eq!(details, vec!["the ".to_string(), " fox".to_string()]);
    }

    #[test]
    fn insert_without_match_creates_single_dendrite() {
        let mut network = NeuralNetwork::new();
        network.insert("alpha", "en");

        assert_eq!(network.dendrites.len(), 1);
        let inserted = network
            .dendrites
            .values()
            .find(|d| d.data == "alpha")
            .expect("expected inserted alpha dendrite");
        assert!(inserted.connections.is_empty());
    }

    #[test]
    fn insert_with_complete_existing_path_makes_no_new_nodes_and_increments_weight() {
        let mut network = seeded_network(&[("hello_uid", "hello"), ("world_uid", "world")]);
        {
            let hello = network
                .dendrites
                .get_mut("hello_uid")
                .expect("expected hello dendrite");
            hello.connect("world_uid".to_string(), 1);
        }

        network.insert("hello world", "en");

        assert_eq!(network.dendrites.len(), 2);
        let hello = network
            .dendrites
            .get("hello_uid")
            .expect("expected hello dendrite");
        let edge = hello
            .connections
            .iter()
            .find(|conn| conn.to == "world_uid")
            .expect("expected hello -> world edge");
        assert_eq!(edge.weight, 2);
    }

    #[test]
    fn insert_adds_missing_tokens_and_connections_for_phrase() {
        let mut network = seeded_network(&[("hello_uid", "hello")]);
        network.insert("hello world", "en");

        let hello = network
            .dendrites
            .get("hello_uid")
            .expect("expected hello dendrite");
        let world = network
            .dendrites
            .values()
            .find(|d| d.data == "world")
            .expect("expected world dendrite");

        assert_eq!(network.dendrites.len(), 2);
        assert!(hello
            .connections
            .iter()
            .any(|conn| conn.from == "hello_uid" && conn.to == world.uid));
    }

    #[test]
    fn insert_compound_word_is_treated_as_distinct_token() {
        let mut network = seeded_network(&[("sun_uid", "sun"), ("filters_uid", "filters")]);
        network.insert("sunlight filters", "en");

        let sunlight = network
            .dendrites
            .values()
            .find(|d| d.data == "sunlight")
            .expect("expected sunlight dendrite to be created as a distinct token");

        assert!(!network.dendrites.values().any(|d| d.data == "light"));
        assert!(sunlight
            .connections
            .iter()
            .any(|conn| conn.from == sunlight.uid && conn.to == "filters_uid"));
    }

    #[test]
    fn insert_stop_words_do_not_reuse_single_global_node() {
        let mut network = NeuralNetwork::new();
        network.insert("the cat", "en");
        network.insert("the dog", "en");

        let the_count = network
            .dendrites
            .values()
            .filter(|d| d.data == "the")
            .count();

        assert_eq!(the_count, 2);
    }

    #[test]
    fn insert_reuses_existing_stop_word_edge_for_same_predecessor() {
        let mut network = NeuralNetwork::new();
        network.insert("through the valley", "en");
        network.insert("through the leaves", "en");

        let through = network
            .dendrites
            .values()
            .find(|d| d.data == "through")
            .expect("expected through dendrite");

        let to_the_count = through
            .connections
            .iter()
            .filter(|conn| {
                network
                    .dendrites
                    .get(&conn.to)
                    .map(|target| target.data == "the")
                    .unwrap_or(false)
            })
            .count();

        assert_eq!(to_the_count, 1);
    }

    #[test]
    fn insert_increments_weight_on_repeated_traversal() {
        let mut network = NeuralNetwork::new();
        network.insert("hello world", "en");
        network.insert("hello world", "en");

        let hello = network
            .dendrites
            .values()
            .find(|d| d.data == "hello")
            .expect("expected hello dendrite");

        let edge = hello
            .connections
            .iter()
            .find(|conn| {
                network
                    .dendrites
                    .get(&conn.to)
                    .map(|target| target.data == "world")
                    .unwrap_or(false)
            })
            .expect("expected hello -> world edge");

        assert_eq!(edge.weight, 2);
    }

    #[test]
    fn enumerate_children_returns_instance_children() {
        let mut network = NeuralNetwork::new();
        network.insert("the mountain stands tall", "en");

        let children = network.enumerate_children("mountain");
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].data, "stands");
    }

    #[test]
    fn save_and_load_round_trip_binary() {
        let mut network = NeuralNetwork::new();
        network.insert("alpha beta", "en");

        let mut path: PathBuf = std::env::temp_dir();
        path.push("neuron_round_trip_test.nrn");
        let filename = path.to_string_lossy().to_string();

        network.save(&filename);
        let mut loaded = NeuralNetwork::new();
        loaded.load(&filename);

        assert_eq!(loaded.all_dendrites_sorted().len(), network.all_dendrites_sorted().len());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn load_legacy_json_is_still_supported() {
        let mut network = NeuralNetwork::new();
        network.insert("legacy format", "en");

        let mut path: PathBuf = std::env::temp_dir();
        path.push("neuron_legacy_round_trip_test.json");
        let filename = path.to_string_lossy().to_string();

        let json = serde_json::to_vec(&network).expect("expected to serialize json");
        std::fs::write(&filename, json).expect("expected to write legacy json");

        let mut loaded = NeuralNetwork::new();
        assert!(loaded.load(&filename));
        assert_eq!(loaded.all_dendrites_sorted().len(), network.all_dendrites_sorted().len());

        let _ = std::fs::remove_file(path);
    }

}