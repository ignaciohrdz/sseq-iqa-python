import cv2
import pickle
from pathlib import Path


if __name__ == "__main__":

    path_image_original = Path("images/test_image_orig.jpg")
    path_image_distorted = Path("images/test_image_dist.jpg")
    path_model = Path("models/csiq/csiq_sseq.pkl")
    with open(path_model, "rb") as f:
        estimator = pickle.load(f)

    img_original = cv2.imread(str(path_image_original))
    img_distorted = cv2.imread(str(path_image_distorted))
    score_original = estimator(img_original)[0]
    score_distorted = estimator(img_distorted)[0]
    print("Original: {:.5f}, Distorted: {:.5f}".format(score_original, score_distorted))
