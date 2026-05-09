import json
import os
from datetime import datetime


def log_experiment(
    log_path,
    name,
    model,
    dataset,
    config,
    hyperparameters,
    metrics, execution_time_mins,
    exp_id=None,
    overwrite=False
):

    entry = {
        "id": exp_id if exp_id else f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "name": name,
        "execution_time_mins": execution_time_mins,
        "model": model,
        "dataset": dataset,
        "config": config,
        "hyperparameters": hyperparameters,
        "metrics": metrics
    }

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            data = json.load(f)
    else:
        data = {"experiments": []}

    if overwrite and exp_id:
        updated = False
        for i, exp in enumerate(data["experiments"]):
            if exp["id"] == exp_id:
                data["experiments"][i] = entry
                updated = True
                break
        if not updated:
            data["experiments"].append(entry)
    else:
        data["experiments"].append(entry)

    with open(log_path, "w") as f:
        json.dump(data, f, indent=4)

    return entry["id"]


# log_experiment(
#     log_path="experiments.json",
#     model="ViT-L/14",
#     dataset="eval",
#     config={
#         "query_expansion": True,
#         "fusion": True,
#         "reranking": True
#     },
#     hyperparameters={
#         "beta": 0.3,
#         "top_k": 50
#     },
#     metrics={
#         "MRR": 0.41,
#         "Recall@1": 0.29,
#         "Recall@5": 0.54
#     }
# )
