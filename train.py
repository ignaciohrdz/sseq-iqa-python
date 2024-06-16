"""
Train a SVR on the several IQA datasets
"""
import pandas as pd

from sseq import SSEQ
import cv2

from data import *
import argparse
from pathlib import Path
import pickle

import random
random.seed(420)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path_datasets", type=str,
                        help="Path to the directory with all datasets")
    parser.add_argument("-d", "--use_dataset", type=str,
                        help="Dataset name: {}".format(list(dataset_fn_dict.keys())))
    parser.add_argument("-m", "--path_model", type=str, default="models",
                        help="Location of the trained SVR models")
    args = parser.parse_args()

    # Define the score without a regressor to obtain the features
    estimator = SSEQ()
    path_model = Path(args.path_model) / args.use_dataset
    path_model.mkdir(exist_ok=True, parents=True)

    path_dataset = Path(args.path_datasets) / args.use_dataset
    path_feature_db = path_dataset / "feature_db.csv"
    if not path_feature_db.exists():
        feature_db = []
        dataset = dataset_fn_dict[args.use_dataset](path_dataset)
        for i, row in enumerate(dataset.to_dict("records")):  # this is much faster than iterrows()
            print("[{}/{}]: Processing {}".format(i+1, len(dataset), row["image_name"]))
            img = cv2.imread(row["image_path"])
            features = estimator(img)
            feature_db.append([row["image_name"]] + features.tolist() + [row["score"]])
        feature_db = pd.DataFrame(feature_db, columns=["image_name"] + list(range(1, 13)) + ["MOS"])
        feature_db.to_csv(path_feature_db, index=False)
    else:
        print("Found feature database. Loading...")
        feature_db = pd.read_csv(path_feature_db).fillna(0)

    print("Fitting an SVR model...")
    results = estimator.fit_svr(feature_db)
    results = pd.DataFrame(results)
    results.to_csv(path_model / "{}_results.csv".format(args.use_dataset), index=False)
    path_model_file = path_model / "{}_sseq.pkl".format(args.use_dataset)
    print("Saving SVR model to ", str(path_model_file))
    with open(path_model_file, "wb") as f:
        pickle.dump(estimator, f, protocol=pickle.HIGHEST_PROTOCOL)
