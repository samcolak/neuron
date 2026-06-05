use crate::helpers::neuralnet::NeuralNetwork;
use crate::helpers::nodenet::{
    NetworkNode, NodeMetadata, NodeNetwork, NodeNetworkController, TokenClusterKeyStrategy,
};
use crate::helpers::text_dendrite::DendriteType;
use crate::helpers::text_similarity::{evaluate_fuzziness, normalize_for_fuzzy_comparison};

use serde::{Serialize, de::DeserializeOwned};
use std::collections::HashSet;
use std::env;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Default)]
pub struct TextNodeController;

#[derive(Debug, Clone, Copy, Default)]
pub struct FirstHitMetrics {
    pub attempted: u64,
    pub accepted: u64,
    pub fallback: u64,
}

static FIRST_HIT_ATTEMPTED: AtomicU64 = AtomicU64::new(0);
static FIRST_HIT_ACCEPTED: AtomicU64 = AtomicU64::new(0);
static FIRST_HIT_FALLBACK: AtomicU64 = AtomicU64::new(0);

pub fn get_first_hit_metrics() -> FirstHitMetrics {
    FirstHitMetrics {
        attempted: FIRST_HIT_ATTEMPTED.load(Ordering::Relaxed),
        accepted: FIRST_HIT_ACCEPTED.load(Ordering::Relaxed),
        fallback: FIRST_HIT_FALLBACK.load(Ordering::Relaxed),
    }
}

fn stop_words_for_language(language: &str) -> Vec<&'static str> {
    match language {
        "en" => vec![
            "a", "an", "and", "as", "at", "by", "for", "from", "in", "is", "of", "on", "or",
            "that", "the", "this", "to", "with",
        ],

        // extension for other languages can be added here in the future
        _ => Vec::new(),
    }
}

fn first_hit_min_candidates() -> usize {
    env::var("NEURON_FIRST_HIT_MIN_CANDIDATES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(1)
}

fn first_hit_min_edge_weight() -> i64 {
    env::var("NEURON_FIRST_HIT_MIN_EDGE_WEIGHT")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(1)
}

impl NodeNetworkController for TextNodeController {
    type Content = str;

    fn tokenize(&self, content: &Self::Content) -> Vec<String> {
        content
            .split_whitespace()
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .map(|token| token.to_string())
            .collect()
    }

    fn normalize_token(&self, token: &str) -> String {
        normalize_for_fuzzy_comparison(token)
    }

    fn evaluate_match(&self, left: &str, right: &str) -> (f64, Vec<String>) {
        evaluate_fuzziness(left, right)
    }

    fn stop_words(&self, metadata: &NodeMetadata) -> Vec<&'static str> {
        let language = metadata.get("lang").unwrap_or("");
        stop_words_for_language(language)
    }
}

impl TokenClusterKeyStrategy for TextNodeController {
    fn cluster_key_for_token(&self, token_key: &str) -> Option<String> {
        let chars: Vec<char> = token_key.chars().filter(|ch| !ch.is_whitespace()).collect();

        if chars.is_empty() {
            return None;
        }

        let first = chars[0];
        let last = chars[chars.len() - 1];
        let len_bucket = chars.len().min(32);

        Some(format!("{}:{}:{}", first, last, len_bucket))
    }
}

impl<N> NodeNetwork<TextNodeController> for NeuralNetwork<TextNodeController, N>
where
    N: NetworkNode + Clone + Serialize + DeserializeOwned,
{
    type Node = N;

    fn enumerate_path_content(&self, content: &str) -> (Option<N>, Vec<N>) {
        let path_tokens: Vec<String> = self
            .tokenize_content(content)
            .into_iter()
            .map(|token| self.token_key_for_index(&token))
            .filter(|token| !token.is_empty())
            .collect();

        if path_tokens.is_empty() {
            return (None, Vec::new());
        }

        let mut current_uids = self.candidate_uids_for_token_vec(&path_tokens[0]);

        for segment_key in &path_tokens[1..] {
            let target_uids = self.candidate_uids_for_token_vec(segment_key);

            if target_uids.is_empty() {
                return (None, Vec::new());
            }

            let target_uid_set: HashSet<&str> = target_uids.iter().map(String::as_str).collect();
            let mut next_uids = Vec::new();

            for uid in &current_uids {
                let Some(dendrite) = self.dendrites().get(uid) else {
                    continue;
                };

                for connection in dendrite.connections() {
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

        if let Some(last_uid) = current_uids.last()
            && let Some(last) = self.dendrites().get(last_uid)
        {
            let mut optional_path = Vec::new();
            for connection in last.connections() {
                if let Some(next) = self.dendrites().get(&connection.to) {
                    optional_path.push(next.clone());
                }
            }
            return (Some(last.clone()), optional_path);
        }

        (None, Vec::new())
    }

    fn insert_content(
        &mut self,
        content: &str,
        metadata: &NodeMetadata,
        dendrite_type: DendriteType,
    ) {
        self.ensure_token_index();

        fn pick_best_uid<N>(
            neural_net: &NeuralNetwork<TextNodeController, N>,
            token_key: &str,
            previous_uid: Option<&str>,
            next_token_key: Option<&str>,
        ) -> Option<String>
        where
            N: NetworkNode + Clone + Serialize + DeserializeOwned,
        {
            let candidates = neural_net.candidate_uids_for_token_vec(token_key);
            if candidates.is_empty() {
                return None;
            }

            let mut attempted_first_hit = false;

            if let Some(prev) = previous_uid
                && candidates.len() >= first_hit_min_candidates()
            {
                attempted_first_hit = true;
                FIRST_HIT_ATTEMPTED.fetch_add(1, Ordering::Relaxed);
                let min_weight = first_hit_min_edge_weight();

                if let Some(uid) =
                    neural_net.best_connected_candidate_uid(prev, &candidates, min_weight)
                {
                    FIRST_HIT_ACCEPTED.fetch_add(1, Ordering::Relaxed);
                    return Some(uid);
                }
            }

            if attempted_first_hit {
                FIRST_HIT_FALLBACK.fetch_add(1, Ordering::Relaxed);
            }

            let next_candidates = next_token_key
                .map(|t| neural_net.candidate_uids_for_token_vec(t))
                .filter(|c| !c.is_empty());

            candidates
                .iter()
                .max_by_key(|candidate_uid| {
                    let mut score = 0;
                    if let Some(prev) = previous_uid
                        && neural_net.has_direct_connection(prev, candidate_uid)
                    {
                        score += 2;
                    }
                    if let Some(next) = next_candidates.as_ref()
                        && next.iter().any(|next_uid| {
                            neural_net.has_direct_connection(candidate_uid, next_uid)
                        })
                    {
                        score += 1;
                    }
                    score
                })
                .cloned()
        }

        let tokens = self.tokenize_content(content);

        if tokens.is_empty() {
            return;
        }

        let token_keys: Vec<String> = tokens
            .iter()
            .map(|token| self.token_key_for_index(token))
            .collect();

        let stop_word_set: HashSet<&'static str> =
            self.controller().stop_words(metadata).into_iter().collect();
        let is_stop_word = |token_key: &str| stop_word_set.contains(token_key);

        let neural_net = self;

        let mut selected_existing_path = Vec::new();
        let mut previous_uid: Option<String> = None;

        for index in 0..tokens.len() {
            if is_stop_word(&token_keys[index]) {
                selected_existing_path.clear();
                break;
            }

            let next_token = token_keys.get(index + 1).map(String::as_str);
            let selected = pick_best_uid(
                neural_net,
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
                .all(|pair| neural_net.has_direct_connection(&pair[0], &pair[1]));

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
                if let Some(previous_uid) = chosen_path.last()
                    && let Some(existing_stop_uid) =
                        neural_net.connected_uid_by_key(previous_uid, &token_keys[index])
                {
                    chosen_path.push(existing_stop_uid);
                    continue;
                }

                let new_uid = neural_net.insert_dendrite_and_index(
                    &tokens[index],
                    metadata,
                    DendriteType::StopWord,
                );
                chosen_path.push(new_uid);

                continue;
            }

            let next_token = token_keys.get(index + 1).map(String::as_str);
            let selected = pick_best_uid(
                neural_net,
                &token_keys[index],
                chosen_path.last().map(String::as_str),
                next_token,
            );

            let uid = match selected {
                Some(existing_uid) => existing_uid,

                None => {
                    neural_net.insert_dendrite_and_index(&tokens[index], metadata, dendrite_type)
                }
            };

            chosen_path.push(uid);
        }

        for pair in chosen_path.windows(2) {
            neural_net.connect_dendrites(&pair[0], &pair[1], 1);
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::helpers::text_dendrite::TextDendrite;

    type TextTestNetwork = NeuralNetwork<TextNodeController, TextDendrite>;

    fn seeded_network(entries: &[(&str, &str)]) -> TextTestNetwork {
        let mut network = TextTestNetwork::new();
        for (uid, data) in entries {
            let mut dendrite =
                TextDendrite::new(data, &NodeMetadata::with_lang("en"), DendriteType::Token);
            dendrite.uid = (*uid).to_string();
            network.dendrites_mut().insert((*uid).to_string(), dendrite);
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
    fn evaluate_fuzziness_handles_minor_typos() {
        let (score, details) = evaluate_fuzziness("neuron", "neurron");
        assert!(score > 0.62);
        assert_eq!(details.len(), 1);
        assert!(details[0].starts_with("fuzzy("));
    }

    #[test]
    fn evaluate_fuzziness_rejects_weighted_matching_for_short_tokens() {
        let (score, _) = evaluate_fuzziness("at", "it");
        assert_eq!(score, 0.0);
    }

    #[test]
    fn evaluate_fuzziness_normalizes_punctuation_and_whitespace() {
        let (score, details) = evaluate_fuzziness("Hello,   world!", "hello world");
        assert_eq!(score, 1.0);
        assert!(details.is_empty());
    }

    #[test]
    fn enumerate_path_can_use_clustered_token_fallback() {
        let mut network = TextTestNetwork::new();
        network.insert(
            "hello world",
            &NodeMetadata::with_lang("en"),
            DendriteType::Token,
        );

        let (node, optional_path) = network.enumerate_path("hello wurld");

        assert!(node.is_some());
        let node = node.expect("expected clustered fallback match");
        assert_eq!(node.data, "world");
        assert!(optional_path.is_empty());
    }

    #[test]
    fn insert_keeps_map_key_and_dendrite_uid_in_sync() {
        let mut network = TextTestNetwork::new();
        network.insert(
            "alpha beta gamma",
            &NodeMetadata::with_lang("en"),
            DendriteType::Token,
        );

        assert!(
            network
                .dendrites()
                .iter()
                .all(|(uid, dendrite)| uid == &dendrite.uid)
        );
    }

    #[test]
    fn insert_without_match_creates_single_dendrite() {
        let mut network = TextTestNetwork::new();
        network.insert("alpha", &NodeMetadata::with_lang("en"), DendriteType::Token);

        assert_eq!(network.dendrites().len(), 1);
        let inserted = network
            .dendrites()
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
                .dendrites_mut()
                .get_mut("hello_uid")
                .expect("expected hello dendrite");
            hello.connect("world_uid".to_string(), 1);
        }

        network.insert(
            "hello world",
            &NodeMetadata::with_lang("en"),
            DendriteType::Token,
        );

        assert_eq!(network.dendrites().len(), 2);
        let hello = network
            .dendrites()
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
        network.insert(
            "hello world",
            &NodeMetadata::with_lang("en"),
            DendriteType::Token,
        );

        let hello = network
            .dendrites()
            .get("hello_uid")
            .expect("expected hello dendrite");
        let world = network
            .dendrites()
            .values()
            .find(|d| d.data == "world")
            .expect("expected world dendrite");

        assert_eq!(network.dendrites().len(), 2);
        assert!(
            hello
                .connections
                .iter()
                .any(|conn| conn.from == "hello_uid" && conn.to == world.uid)
        );
    }

    #[test]
    fn insert_compound_word_is_treated_as_distinct_token() {
        let mut network = seeded_network(&[("sun_uid", "sun"), ("filters_uid", "filters")]);
        network.insert(
            "sunlight filters",
            &NodeMetadata::with_lang("en"),
            DendriteType::Statement,
        );

        let sunlight = network
            .dendrites()
            .values()
            .find(|d| d.data == "sunlight")
            .expect("expected sunlight dendrite to be created as a distinct token");

        assert!(!network.dendrites().values().any(|d| d.data == "light"));
        assert!(
            sunlight
                .connections
                .iter()
                .any(|conn| conn.from == sunlight.uid && conn.to == "filters_uid")
        );
    }

    #[test]
    fn insert_stop_words_do_not_reuse_single_global_node() {
        let mut network = TextTestNetwork::new();
        network.insert(
            "the cat",
            &NodeMetadata::with_lang("en"),
            DendriteType::Statement,
        );
        network.insert(
            "the dog",
            &NodeMetadata::with_lang("en"),
            DendriteType::Statement,
        );

        let the_count = network
            .dendrites()
            .values()
            .filter(|d| d.data == "the")
            .count();

        assert_eq!(the_count, 2);
    }

    #[test]
    fn insert_reuses_existing_stop_word_edge_for_same_predecessor() {
        let mut network = TextTestNetwork::new();
        network.insert(
            "through the valley",
            &NodeMetadata::with_lang("en"),
            DendriteType::Statement,
        );
        network.insert(
            "through the leaves",
            &NodeMetadata::with_lang("en"),
            DendriteType::Statement,
        );

        let through = network
            .dendrites()
            .values()
            .find(|d| d.data == "through")
            .expect("expected through dendrite");

        let to_the_count = through
            .connections
            .iter()
            .filter(|conn| {
                network
                    .dendrites()
                    .get(&conn.to)
                    .map(|target| target.data == "the")
                    .unwrap_or(false)
            })
            .count();

        assert_eq!(to_the_count, 1);
    }

    #[test]
    fn insert_increments_weight_on_repeated_traversal() {
        let mut network = TextTestNetwork::new();
        network.insert(
            "hello world",
            &NodeMetadata::with_lang("en"),
            DendriteType::Statement,
        );
        network.insert(
            "hello world",
            &NodeMetadata::with_lang("en"),
            DendriteType::Statement,
        );

        let hello = network
            .dendrites()
            .values()
            .find(|d| d.data == "hello")
            .expect("expected hello dendrite");

        let edge = hello
            .connections
            .iter()
            .find(|conn| {
                network
                    .dendrites()
                    .get(&conn.to)
                    .map(|target| target.data == "world")
                    .unwrap_or(false)
            })
            .expect("expected hello -> world edge");

        assert_eq!(edge.weight, 2);
    }

    #[test]
    fn enumerate_children_returns_instance_children() {
        let mut network = TextTestNetwork::new();
        network.insert(
            "the mountain stands tall",
            &NodeMetadata::with_lang("en"),
            DendriteType::Statement,
        );

        let children = network.enumerate_children("mountain");
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].data, "stands");
    }
}
