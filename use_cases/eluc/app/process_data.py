"""
Pre-downloads the dataset file from the Hugging Face Hub repository.
"""
from huggingface_hub import hf_hub_download


if __name__ == "__main__":
    hf_hub_download(repo_id="projectresilience/land-use-app-data",
                    filename="app_data.csv",
                    local_dir="app/data",
                    repo_type="dataset")
