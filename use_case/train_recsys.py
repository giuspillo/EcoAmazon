import ray
from ray import tune
from recbole.quick_start import run_recbole
import os
import torch


def train_recbole(config, model_name, dataset_name, exp_name):
    """Train a given model on a specific dataset with Ray."""
    # Setup
    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_path, "dataset")
    config_str = f"lr_{config['lr']:.4f}_emb_{config['embedding_size']}_reg_{config['reg_weight']:.5f}"
    unique_checkpoint_dir = os.path.join(base_path, "models", exp_name, config_str)

    # Run training
    result = run_recbole(
        model=model_name,
        dataset=dataset_name,
        config_dict={
            "tensorboard": False,
            "device": device,
            "epochs": config["epochs"],
            "eval_step": 10,
            "learning_rate": config["lr"],
            "embedding_size": config["embedding_size"],
            "reg_weight": config["reg_weight"],
            **({"n_layers": config["n_layers"]} if model_name == "LightGCN" else {}),
            "benchmark_filename": ["train", "valid", "test"],
            "data_path": dataset_path,
            "checkpoint_dir": unique_checkpoint_dir,
        },
    )

    # RecBole returns best valid score dict
    valid_recall = result["best_valid_score"]
    tune.report({"recall_10": valid_recall})


def run_hpo(model_name, num_samples):
    """Run HPO for a specific model."""
    # Common hyperparameters
    search_space = {
        "lr": tune.loguniform(1e-4, 5e-3),
        "embedding_size": tune.choice([32, 64]),
        "reg_weight": tune.loguniform(1e-5, 1e-4),
        "epochs": 100,
    }

    # Light GCN-specific hyperparameters
    if model_name == "LightGCN":
        search_space["n_layers"] = tune.choice([1, 2, 3])

    # Run HPO
    analysis = tune.run(
        tune.with_parameters(
            train_recbole,
            model_name=model_name,
            dataset_name="amazon_elec",
        ),
        metric="recall_10",
        mode="max",
        config=search_space,
        num_samples=num_samples,
        resources_per_trial={"cpu": 4},
    )

    # Get and return the best hyperparameters found
    best_config = analysis.get_best_config(metric="recall_10", mode="max")
    return best_config


# =============================================
# Setup
# =============================================
# Configure device and initialize Ray
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ray.init(include_dashboard=True, dashboard_host="127.0.0.1", dashboard_port=8265)

# =============================================
# HPO both for BPR and Light GCN
# =============================================
# Run HPO both for BPR and Light GCN
best_bpr = run_hpo("BPR", num_samples=10)
best_lgcn = run_hpo("LightGCN", num_samples=15)

# =============================================
# Final training both for BPR and Light GCN
# =============================================
# Models to train
models_to_train = {"BPR": best_bpr, "LightGCN": best_lgcn}

# Retrain the models with their best
# hyperparameters found
for model_name, hyperparams in models_to_train.items():
    run_recbole(
        model=model_name,
        dataset="amazon_elec",
        config_dict={
            **hyperparams,
            "device": device,
            "benchmark_filename": ["train", "valid", "test"],
            "checkpoint_dir": f"./models/{model_name}_best",
        },
    )