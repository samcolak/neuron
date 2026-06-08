use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use neuralnet::core::brain::MultiModalNeuralNetwork;
use neuralnet::core::nodenet::NodeMetadata;
use neuralnet::dendrites::text_dendrite::DendriteType;
use neuralnet::helpers::multimodal_controller::MultiModalInput;

#[derive(Debug, Clone, Copy)]
struct StressConfig {
    insert_count: usize,
    query_rounds: usize,
    snapshot_every: usize,
}

#[derive(Debug, Clone)]
struct StressReport {
    scenario: &'static str,
    insert_count: usize,
    query_rounds: usize,
    insert_ms: u128,
    query_ms: u128,
    total_nodes: usize,
    average_query_score: f64,
    snapshot_enabled: bool,
    snapshot_pending: usize,
    snapshot_error: Option<String>,
    snapshot_files_present: bool,
}

fn parse_env_usize(name: &str, default_value: usize) -> usize {
    match env::var(name) {
        Ok(value) => value
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|parsed| *parsed > 0)
            .unwrap_or(default_value),
        Err(_) => default_value,
    }
}

fn stress_snapshot_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("brain_stress_walkthrough")
}

fn run_stress_scenario(config: StressConfig, enable_snapshot: bool) -> StressReport {
    let mut network = MultiModalNeuralNetwork::new_multimodal();
    let metadata = NodeMetadata::with_lang("en");

    let mut snapshot_files_present = false;
    let mut snapshot_pending = 0usize;
    let mut snapshot_error = None;

    if enable_snapshot {
        let snapshot_dir = stress_snapshot_dir();
        let _ = fs::create_dir_all(snapshot_dir.as_path());
        let snapshot_id = "brain_stress_walkthrough";
        network.enable_auto_snapshot_in_dir(snapshot_id, snapshot_dir.as_path(), config.snapshot_every);

        let bundle_path =
            network.snapshot_bundle_path_for_instance_in_dir(snapshot_id, snapshot_dir.as_path());

        let insert_start = Instant::now();
        for idx in 0..config.insert_count {
            let sentence = format!("stress token {:06} bucket {:03}", idx, idx % 113);
            if idx % 5 == 0 {
                network.absorb_true_text(sentence.as_str(), &metadata, DendriteType::Statement);
            } else {
                network.insert_text(sentence.as_str(), &metadata, DendriteType::Statement);
            }
        }
        let insert_ms = insert_start.elapsed().as_millis();

        let _ = network.flush_auto_snapshot();
        snapshot_pending = network.auto_snapshot_pending_inserts();
        snapshot_error = network.auto_snapshot_last_error().map(str::to_string);
        snapshot_files_present = bundle_path.exists();

        let query_start = Instant::now();
        let mut score_sum = 0.0;
        for round in 0..config.query_rounds {
            let idx = round % config.insert_count;
            let query = MultiModalInput::Text(format!("stress token {:06}", idx));
            score_sum += network.evaluate_question_fuzziness(&query);
        }
        let query_ms = query_start.elapsed().as_millis();

        return StressReport {
            scenario: "with_batched_snapshot",
            insert_count: config.insert_count,
            query_rounds: config.query_rounds,
            insert_ms,
            query_ms,
            total_nodes: network.all_dendrites_sorted().len(),
            average_query_score: score_sum / config.query_rounds as f64,
            snapshot_enabled: true,
            snapshot_pending,
            snapshot_error,
            snapshot_files_present,
        };
    }

    let insert_start = Instant::now();
    for idx in 0..config.insert_count {
        let sentence = format!("stress token {:06} bucket {:03}", idx, idx % 113);
        if idx % 5 == 0 {
            network.absorb_true_text(sentence.as_str(), &metadata, DendriteType::Statement);
        } else {
            network.insert_text(sentence.as_str(), &metadata, DendriteType::Statement);
        }
    }
    let insert_ms = insert_start.elapsed().as_millis();

    let query_start = Instant::now();
    let mut score_sum = 0.0;
    for round in 0..config.query_rounds {
        let idx = round % config.insert_count;
        let query = MultiModalInput::Text(format!("stress token {:06}", idx));
        score_sum += network.evaluate_question_fuzziness(&query);
    }
    let query_ms = query_start.elapsed().as_millis();

    StressReport {
        scenario: "in_memory_only",
        insert_count: config.insert_count,
        query_rounds: config.query_rounds,
        insert_ms,
        query_ms,
        total_nodes: network.all_dendrites_sorted().len(),
        average_query_score: score_sum / config.query_rounds as f64,
        snapshot_enabled: false,
        snapshot_pending,
        snapshot_error,
        snapshot_files_present,
    }
}

pub fn run_brain_stress_walkthrough() {
    let config = StressConfig {
        insert_count: parse_env_usize("NEURON_BRAIN_STRESS_INSERTS", 2_000),
        query_rounds: parse_env_usize("NEURON_BRAIN_STRESS_QUERIES", 500),
        snapshot_every: parse_env_usize("NEURON_BRAIN_STRESS_SNAPSHOT_EVERY", 250),
    };

    println!("\nBrain stress walkthrough");
    println!(
        "  config: inserts={} queries={} snapshot_every={}",
        config.insert_count, config.query_rounds, config.snapshot_every
    );

    let in_memory_report = run_stress_scenario(config, false);
    println!(
        "  scenario={} insert_ms={} query_ms={} nodes={} avg_score={:.3}",
        in_memory_report.scenario,
        in_memory_report.insert_ms,
        in_memory_report.query_ms,
        in_memory_report.total_nodes,
        in_memory_report.average_query_score,
    );

    let snapshot_report = run_stress_scenario(config, true);
    println!(
        "  scenario={} insert_ms={} query_ms={} nodes={} avg_score={:.3}",
        snapshot_report.scenario,
        snapshot_report.insert_ms,
        snapshot_report.query_ms,
        snapshot_report.total_nodes,
        snapshot_report.average_query_score,
    );
    println!(
        "    snapshot_enabled={} files_present={} pending={} last_error={}",
        snapshot_report.snapshot_enabled,
        snapshot_report.snapshot_files_present,
        snapshot_report.snapshot_pending,
        snapshot_report
            .snapshot_error
            .as_deref()
            .unwrap_or("<none>")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stress_scenario_runs_without_snapshot() {
        let report = run_stress_scenario(
            StressConfig {
                insert_count: 200,
                query_rounds: 40,
                snapshot_every: 50,
            },
            false,
        );

        assert!(!report.snapshot_enabled);
        assert!(report.total_nodes > 0);
        assert_eq!(report.insert_count, 200);
        assert_eq!(report.query_rounds, 40);
    }

    #[test]
    fn stress_scenario_runs_with_batched_snapshot() {
        let report = run_stress_scenario(
            StressConfig {
                insert_count: 200,
                query_rounds: 40,
                snapshot_every: 100,
            },
            true,
        );

        assert!(report.snapshot_enabled);
        assert!(report.snapshot_files_present);
        assert_eq!(report.snapshot_pending, 0);
        assert!(report.snapshot_error.is_none());
    }
}
