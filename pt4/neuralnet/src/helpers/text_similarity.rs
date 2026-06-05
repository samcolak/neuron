use std::collections::HashMap;

fn normalized_levenshtein(a: &str, b: &str) -> f64 {
    
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    if a_chars.is_empty() && b_chars.is_empty() {
        return 1.0;
    }

    if a_chars.is_empty() || b_chars.is_empty() {
        return 0.0;
    }

    let mut previous_row: Vec<usize> = (0..=b_chars.len()).collect();
    let mut current_row = vec![0; b_chars.len() + 1];

    for (i, a_char) in a_chars.iter().enumerate() {
        current_row[0] = i + 1;

        for (j, b_char) in b_chars.iter().enumerate() {
            let substitution_cost = if a_char == b_char { 0 } else { 1 };

            current_row[j + 1] = (previous_row[j + 1] + 1)
                .min(current_row[j] + 1)
                .min(previous_row[j] + substitution_cost);
        }

        std::mem::swap(&mut previous_row, &mut current_row);
    }

    let distance = previous_row[b_chars.len()] as f64;
    let max_len = a_chars.len().max(b_chars.len()) as f64;
    (1.0 - (distance / max_len)).clamp(0.0, 1.0)

}

fn character_bigram_dice_similarity(a: &str, b: &str) -> f64 {

    fn bigrams(input: &str) -> Vec<(char, char)> {
        let chars: Vec<char> = input.chars().collect();
        chars.windows(2).map(|w| (w[0], w[1])).collect()
    }

    let a_bigrams = bigrams(a);
    let b_bigrams = bigrams(b);

    if a_bigrams.is_empty() && b_bigrams.is_empty() {
        return 1.0;
    }

    if a_bigrams.is_empty() || b_bigrams.is_empty() {
        return 0.0;
    }

    let mut a_counts: HashMap<(char, char), usize> = HashMap::new();
    let mut b_counts: HashMap<(char, char), usize> = HashMap::new();

    for gram in a_bigrams {
        *a_counts.entry(gram).or_insert(0) += 1;
    }

    for gram in b_bigrams {
        *b_counts.entry(gram).or_insert(0) += 1;
    }

    let shared = a_counts
        .iter()
        .map(|(gram, a_count)| {
            let b_count = b_counts.get(gram).copied().unwrap_or(0);
            (*a_count).min(b_count)
        })
        .sum::<usize>() as f64;

    let total = (a_counts.values().sum::<usize>() + b_counts.values().sum::<usize>()) as f64;

    if total == 0.0 {
        0.0
    } else {
        (2.0 * shared) / total
    }

}

pub fn normalize_for_fuzzy_comparison(input: &str) -> String {

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

pub fn evaluate_fuzziness(content: &str, data: &str) -> (f64, Vec<String>) {

    let contentlower = content.to_ascii_lowercase();
    let datalower = data.to_ascii_lowercase();
    let normalized_content = normalize_for_fuzzy_comparison(content);
    let normalized_data = normalize_for_fuzzy_comparison(data);

    if normalized_content == normalized_data {
        return (1.0, Vec::new());
    }

    if contentlower.contains(&datalower) {
        let mut pieces = Vec::new();
        let index = contentlower.find(&datalower).unwrap_or(0);
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
        let index = datalower.find(&contentlower).unwrap_or(0);
        if index > 0 {
            pieces.push(data[0..index].to_string());
        }

        let end_index = index + contentlower.len();
        if end_index < data.len() {
            pieces.push(data[end_index..].to_string());
        }

        return (0.8, pieces);
    }

    let normalized_content_len = normalized_content.chars().count();
    let normalized_data_len = normalized_data.chars().count();

    if normalized_content_len < 3 || normalized_data_len < 3 {
        return (0.0, vec![content.to_string()]);
    }

    let edit_similarity = normalized_levenshtein(&normalized_content, &normalized_data);
    let bigram_similarity = character_bigram_dice_similarity(&normalized_content, &normalized_data);
    let prefix_similarity = if normalized_content.chars().next() == normalized_data.chars().next() {
        1.0
    } else {
        0.0
    };

    let weighted_similarity =
        ((0.70 * edit_similarity) + (0.20 * bigram_similarity) + (0.10 * prefix_similarity))
            .clamp(0.0, 1.0);

    if weighted_similarity >= 0.60 {
        return (
            weighted_similarity,
            vec![format!(
                "fuzzy(edit={:.3}, bigram={:.3}, prefix={:.0}, score={:.3})",
                edit_similarity, bigram_similarity, prefix_similarity, weighted_similarity,
            )],
        );
    }

    (0.0, vec![content.to_string()])

}
