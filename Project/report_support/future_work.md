# Future Work

- K-fold cross-validation across the full pipeline for a more robust generalization estimate.
- Multi-class modelling of the `Sub_Cat` attack families, not just binary Normal/Anomaly.
- Evaluation on additional or more recent IoT traffic captures to test robustness to concept drift.
- Threshold tuning and cost-sensitive evaluation matched to a specific deployment's false
  positive/false negative cost trade-off.
- Model compression/quantization and benchmarking for deployment on constrained IoT gateway hardware.
- Adversarial robustness testing against evasion attempts targeting the selected features.
- Online/incremental learning to adapt to new attack patterns without full retraining.
