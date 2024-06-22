""" A very simple script to get a table with all models' test metrics """

import pickle
from pathlib import Path
import pandas as pd


if __name__ == "__main__":
    path_models = Path("models")
    results_table = pd.DataFrame(columns=["Dataset", "LCC", "SROCC"])
    for m in path_models.glob("**/*.pkl"):
        with open(m, "rb") as f:
            estimator = pickle.load(f)
            dataset = m.parent.name
            lcc, srocc = estimator.test_results.values()
            results_table.loc[len(results_table), :] = [dataset, lcc, srocc]
    results_table.round(4).to_csv(path_models / "results.csv", index=False)
