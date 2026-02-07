import torch
import pandas as pd
import numpy as np
from recbole.evaluator import Evaluator
from recbole.quick_start import load_data_and_model

from utils import get_latest_checkpoint, get_co2e_kg_estimations


def calculate_average_pcf(reranked_items_list, id_to_asin, scores_dict, dataset, k):
    """Calculate average PCF across all the users based on the first-k recommendations."""
    total_pcf = 0
    count = 0
    for user_items in reranked_items_list:
        for item_id in user_items[:k]:
            token = dataset.id2token(dataset.iid_field, int(item_id))
            asin = id_to_asin.get(int(token))
            pcf = scores_dict.get(asin)
            if pcf is not None:
                total_pcf += pcf
                count += 1
    return total_pcf / count if count > 0 else 0


def evaluate_model_results(data, k, config, dataset, id_to_asin, co2e_scores, item_cnt):
    """Evaluate the RecSys both on RecBole and sustainability metrics, considering
    the top-k recommendations."""
    # Retrieve top-k re-ranked items and prepare tensors
    # for RecBole (considering just the first-k items)
    reranked_items_list = data["reranked_items"]
    reranked_np = np.array(reranked_items_list)[:, :k]
    reranked_items = torch.tensor(reranked_np, device=config["device"])

    pos_matrix = torch.tensor(data["pos_matrix"], device=config["device"])[:, :k]
    pos_len = torch.tensor(data["pos_len"], device=config["device"]).view(-1, 1)

    # Run RecBole evaluation, defining its configuration
    config["topk"] = [k]
    struct = {
        "rec.topk": torch.cat((pos_matrix, pos_len), dim=1).cpu(),
        "rec.items": reranked_items.cpu(),
        "data.num_items": dataset.item_num,
        "data.count_items": item_cnt,
    }
    rec_results = Evaluator(config).evaluate(struct)

    # Calculate sustainability metrics
    avg_pcf_est = calculate_average_pcf(
        reranked_items_list, id_to_asin, co2e_scores, dataset, k
    )

    # Collect and clean results
    res = {
        "MODEL": data["model"],
        "ALPHA": data["alpha"],
        "K": k,
        "PCF": round(avg_pcf_est, 4),
    }
    for m, v in rec_results.items():
        metric_name = m.split("@")[0].upper()
        res[metric_name] = round(v, 4)

    # Calculate F1
    prec = res.get("PRECISION")
    rec = res.get("RECALL")
    if prec is not None and rec is not None:
        if (prec + rec) > 0:
            res["F1"] = round(2 * (prec * rec) / (prec + rec), 4)
        else:
            res["F1"] = 0.0

    return res


# =============================================
# Setup
# =============================================
# "gemini-2_5-flash" or "o3-mini"
model_tag = "o3-mini"
alpha_values = [0.25, 0.5, 0.75, 1.0]
models = ["BPR", "LightGCN"]
k_values = [5, 10, 20, 50]

# Load configuration and dataset
config, _, dataset, *_ = load_data_and_model(
    model_file=get_latest_checkpoint("LightGCN_best"),
)

# Upload configuration to define evaluation metrics
config["metrics"] = [
    "Recall",
    "NDCG",
    "GiniIndex",
    "AveragePopularity",
    "TailPercentage",
    "ItemCoverage",
    "Precision",
    "MRR",
    "Hit",
]

# Load CO2 score estimations (LLM)
co2e_scores = get_co2e_kg_estimations(model_tag)

# Load item_index -> parent_asin mapping
item_map_df = pd.read_csv("../2_recbole/process_data/maps/item_map.tsv", sep="\t")
id_to_asin = dict(zip(item_map_df["item_index"], item_map_df["parent_asin"]))

# For calculating item popularity
train_item_ids = dataset.inter_feat[dataset.iid_field].numpy()
item_cnt_array = np.zeros(dataset.item_num, dtype=np.int64)
for iid in train_item_ids:
    item_cnt_array[iid] += 1
item_cnt = [(i, count) for i, count in enumerate(item_cnt_array)]

# =============================================
# RecSys — Evaluation
# =============================================
# For each RecSys model, for each alpha value,
# and for each different k, evaluate the model
# under RecBole and sustainability metrics
all_final_results = []
for model in models:
    for alpha in alpha_values:
        file_path = (
            f"../3_reranking/results/{model}/{model_tag}/results_alpha_{alpha}.pth"
        )
        data = torch.load(file_path, weights_only=False)
        for k_val in k_values:
            res = evaluate_model_results(
                data, k_val, config, dataset, id_to_asin, co2e_scores, item_cnt
            )
            all_final_results.append(res)

# Save results
df_final = pd.DataFrame(all_final_results)
df_final = df_final.sort_values(["MODEL", "K", "ALPHA"], ascending=[True, True, False])
df_final.to_csv(f"results/{model_tag}/results.csv", index=False)

# Show results
print("\n" + "=" * 150 + "\n RECSYS EVALUATION — RESULTS \n" + "=" * 150)
print(df_final.to_string(index=False))