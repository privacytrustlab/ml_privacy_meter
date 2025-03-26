```
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
      population_signals,
      reuse_offline_a=False
  )
```
