```
# Each run uses one model as target model and selects reference models from the remaining models
target_model_indices = list(range(2*(num_experiments+1)))
reference_model_indices_all = []
for target_model_idx in target_model_indices:
    paired_model_idx = (
        target_model_idx + 1 if target_model_idx % 2 == 0 else target_model_idx - 1
    )
    # Select reference models from non-target and non-paired model indices
    ref_indices = [
        I
        for i in range(signals.shape[1])
        if i != target_model_idx and i != paired_model_idx
    ][: 2 * num_reference_models]
    reference_model_indices_all.append(np.array(ref_indices))

args = {
        "attack": "RMIA",
        "dataset": configs["data"]["dataset"],
        "model": configs["train"]["model_name"],
        "offline_a": 0.3
    }

# Initialize the MIA instance
MIA_instance = MIA(logger)
mia_scores, target_memberships = MIA_instance.run_mia(
      all_signals, 
      all_memberships, 
      target_model_idx, 
      reference_model_indices, 
      logger, 
      args,
      reuse_offline_a=False
  )
```
