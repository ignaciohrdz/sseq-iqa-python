""" This is an example of how to use the models """

import cv2
import pickle
import argparse
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--use_dataset", type=str,
                        help="Dataset used to train the SVR model")
    args = parser.parse_args()

    path_image_original = Path("images/test_image_orig.jpg")
    path_image_distorted = Path("images/test_image_dist.jpg")
    path_model = Path("models/{}/{}_sseq.pkl".format(args.use_dataset, args.use_dataset))
    with open(path_model, "rb") as f:
        estimator = pickle.load(f)

    img_original = cv2.imread(str(path_image_original))
    img_distorted = cv2.imread(str(path_image_distorted))
    score_original = estimator(img_original)[0]
    score_distorted = estimator(img_distorted)[0]
    print("Original: {:.5f}, Distorted: {:.5f}".format(score_original, score_distorted))
