import glob
import json
import os


def get_latest_checkpoint(model_name):
    """Get the latest model checkpoint from its folder."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    saved_model_dir = os.path.join(base_path, "2_recbole", "models", model_name)
    checkpoint_files = glob.glob(os.path.join(saved_model_dir, "*.pth"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No model checkpoint found at '{saved_model_dir}'.")
    latest_file = sorted(checkpoint_files)[-1]
    return latest_file


def get_co2e_kg_estimations(model_tag):
    # Load CO2 score estimations for the given model
    co2e_scores = {}
    with open(
        f"../1_pcf/results/full/{model_tag}/results.jsonl", "r", encoding="utf-8"
    ) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("co2e_kg") is not None:
                co2e_scores[data["parent_asin"]] = data["co2e_kg"]
    return co2e_scores
