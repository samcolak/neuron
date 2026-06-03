use crate::helpers::textdendrite::DendriteType;
use crate::helpers::neuralnet::NeuralNetwork;
use crate::helpers::nodenet::{
    NetworkNode,
    NodeNetwork,
    NodeNetworkController,
    TokenClusterKeyStrategy,
};
use serde::{de::DeserializeOwned, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Default)]
pub struct NgramController;

fn normalize_for_ngram(input: &str) -> String {
    input
        .to_lowercase()
        .chars()
        .filter(|ch| ch.is_alphanumeric())
        .collect()
}

impl TokenClusterKeyStrategy for NgramController {
    fn cluster_key_for_token(&self, token_key: &str) -> Option<String> {
        if token_key.is_empty() {
            return None;
        }

        let chars: Vec<char> = token_key.chars().collect();
        let len_bucket = chars.len().min(32);

        let prefix = if chars.len() >= 2 {
            format!("{}{}", chars[0], chars[1])
        } else {
            chars[0].to_string()
        };

        Some(format!("{}:{}", prefix, len_bucket))
    }
}

impl NodeNetworkController for NgramController {
    type Content = str;

    fn tokenize(&self, content: &Self::Content) -> Vec<String> {
        let normalized = normalize_for_ngram(content);

        if normalized.is_empty() {
            return Vec::new();
        }

        let chars: Vec<char> = normalized.chars().collect();

        if chars.len() <= 3 {
            return vec![normalized];
        }

        chars
            .windows(3)
            .map(|window| window.iter().collect())
            .collect()
    }

    fn normalize_token(&self, token: &str) -> String {
        normalize_for_ngram(token)
    }

    fn evaluate_match(&self, left: &str, right: &str) -> (f64, Vec<String>) {
        if left == right {
            (1.0, Vec::new())
        } else {
            (0.0, vec![left.to_string()])
        }
    }

    fn stop_words(&self, _language: &str) -> Vec<&'static str> {
        Vec::new()
    }
}

impl<N> NodeNetwork<NgramController> for NeuralNetwork<NgramController, N>
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
                let Some(node) = self.dendrites().get(uid) else {
                    continue;
                };

                for connection in node.connections() {
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

    fn insert_content(&mut self, content: &str, language: &str, dendrite_type: DendriteType) {
        self.ensure_token_index();

        let tokens = self.tokenize_content(content);
        if tokens.is_empty() {
            return;
        }

        let mut chosen_path = Vec::new();

        for token in tokens {
            let token_key = self.token_key_for_index(&token);
            if token_key.is_empty() {
                continue;
            }

            let uid = self
                .candidate_uids_for_token_vec(&token_key)
                .into_iter()
                .next()
                .unwrap_or_else(|| self.insert_dendrite_and_index(&token, language, dendrite_type));

            chosen_path.push(uid);
        }

        for pair in chosen_path.windows(2) {
            self.connect_dendrites(&pair[0], &pair[1], 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::textdendrite::TextDendrite;
    use crate::helpers::nodenet::NodeNetwork;

    #[test]
    fn tokenize_produces_character_trigrams() {
        let controller = NgramController;
        let tokens = controller.tokenize("Neural");

        assert_eq!(tokens, vec!["neu", "eur", "ura", "ral"]);
    }

    #[test]
    fn ngram_insert_and_enumerate_path_work() {
        let mut network: NeuralNetwork<NgramController, TextDendrite> =
            NeuralNetwork::with_controller(NgramController);

        network.insert_content("neural", "en", DendriteType::Token);

        let all = network.all_dendrites_sorted();
        assert!(all.len() >= 4);
        assert!(all.iter().any(|node| node.data == "neu"));

        let (last, _) = network.enumerate_path_content("neural");
        assert!(last.is_some());
    }
}
