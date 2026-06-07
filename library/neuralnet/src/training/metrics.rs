use std::collections::{BTreeMap, BTreeSet};

use crate::training::trainer::{LabelQualityMetrics, TrainerEvaluationReport};

pub(crate) fn increment_confusion_count(
    matrix: &mut BTreeMap<String, BTreeMap<String, usize>>,
    expected: String,
    predicted: String,
) {
    *matrix
        .entry(expected)
        .or_default()
        .entry(predicted)
        .or_insert(0) += 1;
}

pub(crate) fn compute_quality_metrics(report: &mut TrainerEvaluationReport) {
    let mut labels: BTreeSet<String> = BTreeSet::new();

    labels.extend(report.per_label_total.keys().cloned());
    for predicted_map in report.confusion_matrix.values() {
        for predicted in predicted_map.keys() {
            if predicted != "<unknown>" {
                labels.insert(predicted.clone());
            }
        }
    }

    if labels.is_empty() {
        return;
    }

    let mut precision_sum = 0.0;
    let mut recall_sum = 0.0;
    let mut f1_sum = 0.0;
    let mut tp_total: usize = 0;
    let mut fp_total: usize = 0;
    let mut fn_total: usize = 0;

    for label in labels {
        let tp = report
            .confusion_matrix
            .get(&label)
            .and_then(|row| row.get(&label))
            .copied()
            .unwrap_or(0);

        let fp = report
            .confusion_matrix
            .iter()
            .filter(|(expected, _)| *expected != &label)
            .map(|(_, row)| row.get(&label).copied().unwrap_or(0))
            .sum::<usize>();

        let fn_count = report
            .confusion_matrix
            .get(&label)
            .map(|row| {
                row.iter()
                    .filter(|(predicted, _)| *predicted != &label)
                    .map(|(_, count)| *count)
                    .sum::<usize>()
            })
            .unwrap_or(0);

        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };

        let recall = if tp + fn_count > 0 {
            tp as f64 / (tp + fn_count) as f64
        } else {
            0.0
        };

        let f1 = if (precision + recall) > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        let support = report.per_label_total.get(&label).copied().unwrap_or(0);

        report.per_label_metrics.insert(
            label,
            LabelQualityMetrics {
                precision,
                recall,
                f1,
                support,
            },
        );

        precision_sum += precision;
        recall_sum += recall;
        f1_sum += f1;
        tp_total += tp;
        fp_total += fp;
        fn_total += fn_count;
    }

    let label_count = report.per_label_metrics.len() as f64;
    if label_count > 0.0 {
        report.macro_precision = precision_sum / label_count;
        report.macro_recall = recall_sum / label_count;
        report.macro_f1 = f1_sum / label_count;
    }

    report.micro_precision = if tp_total + fp_total > 0 {
        tp_total as f64 / (tp_total + fp_total) as f64
    } else {
        0.0
    };

    report.micro_recall = if tp_total + fn_total > 0 {
        tp_total as f64 / (tp_total + fn_total) as f64
    } else {
        0.0
    };

    report.micro_f1 = if (report.micro_precision + report.micro_recall) > 0.0 {
        2.0 * report.micro_precision * report.micro_recall
            / (report.micro_precision + report.micro_recall)
    } else {
        0.0
    };
}