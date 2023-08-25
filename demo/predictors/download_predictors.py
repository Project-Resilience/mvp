import os
import json
from huggingface_hub import hf_hub_download


def main():
    file_name_list = []
    predictor_cfg = json.load(open("../predictors/predictors.json"))
    for row in predictor_cfg["predictors"]:
        file_name_list.append(row["filename"])

    for predictor_name in file_name_list:
        if not os.path.exists(predictor_name):
            hf_hub_download(
                token=os.environ.get("HF_TOKEN"),
                repo_id="danyoung/eluc-dataset",
                repo_type="dataset",
                filename=predictor_name,
                local_dir="./",
                local_dir_use_symlinks=False)


if __name__ == "__main__":
    main()