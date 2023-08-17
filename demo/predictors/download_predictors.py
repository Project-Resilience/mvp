import os
from huggingface_hub import hf_hub_download

PREDICTOR_LIST = ["ELUC_forest.joblib", "ELUC_linreg.joblib"]

def main():
    for predictor_name in PREDICTOR_LIST:
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