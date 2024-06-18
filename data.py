import pandas as pd
from pathlib import Path


def prepare_koniq(path_koniq: Path):
    """ Prepares the KonIQ-10k dataset for training """
    path_images = path_koniq / "1024x768"
    dataset = pd.read_csv(path_koniq / "koniq10k_scores_and_distributions.csv").rename(columns={"MOS": "score"})
    dataset["image_path"] = dataset["image_name"].apply(lambda x: str(path_images / x))
    return dataset.loc[:, ["image_path", "image_name", "score"]]


def prepare_kadid(path_kadid: Path):
    """ Prepares the KADID-10k dataset for training """
    path_images = path_kadid / "images"
    dataset = pd.read_csv(path_kadid / "dmos.csv").rename(columns={"dist_img": "image_name", "dmos": "score"})
    dataset["image_path"] = dataset["image_name"].apply(lambda x: str(path_images / x))
    return dataset.loc[:, ["image_path", "image_name", "score"]]


def prepare_csiq(path_csiq: Path):
    """ Prepares the CSIQ dataset for training """
    path_distorted_images = path_csiq / "dst_imgs"
    dataset = pd.read_excel(path_csiq / "csiq.DMOS.xlsx", sheet_name="all_by_image")
    dataset["image"] = dataset["image"].astype(str)
    dataset = dataset.set_index(["image", "dst_type", "dst_lev"])

    csiq_data = pd.DataFrame(columns=["image_path", "image_name", "score"])
    for f in path_distorted_images.glob("**/*.png"):
        img_name, dst_type, dst_level = str(f.stem).lower().split(".")

        # Distortion name changes
        dst_type = dst_type.replace("awgn", "noise")
        dst_type = dst_type.replace("jpeg2000", "jpeg 2000")

        # For some reason, not all the distorted images were rated,
        # and they don"t appear in the Excel file
        try:
            score = dataset.loc[(img_name, dst_type, int(dst_level)), "dmos"]
            csiq_data.loc[len(csiq_data), :] = [str(f), f.name, score]
        except KeyError:
            print("Score not found for this CSIQ image: ", f.name)
    return csiq_data


def prepare_tid(path_tid: Path):
    """ Prepares the TID2013 dataset for training """
    path_images = path_tid / "distorted_images"
    dataset = pd.read_csv(path_tid / "mos_with_names.txt", names=["score", "image_name"], sep=" ")
    dataset["image_path"] = dataset["image_name"].apply(lambda x: str(path_images / x))
    return dataset.loc[:, ["image_path", "image_name", "score"]]


dataset_fn_dict = {
    "koniq10k": prepare_koniq,
    "kadid10k": prepare_kadid,
    "csiq": prepare_csiq,
    "tid2013": prepare_tid
}