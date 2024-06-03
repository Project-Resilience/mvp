"""
Script to upload a model to huggingface hub.
"""
from argparse import ArgumentParser
from pathlib import Path

from huggingface_hub import HfApi

def write_readme(model_path: str):
    """
    Writes readme to model save path to upload.
    TODO: Need to add more info to the readme and make it a proper template.
    """
    model_path = Path(model_path)
    with open(model_path / "README.md", "w", encoding="utf-8") as file:
        file.write("This is a demo model created for project resilience")

def upload_to_repo(model_path: str, repo_id: str, token: str=None):
    """
    Uses huggingface hub to upload the model to a repo.
    """
    model_path = Path(model_path)
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        token=token
    )

    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_id,
        repo_type="model",
        token=token
    )

def main():
    """
    Main logic for uploading a model.
    """
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--token", type=str, required=False)
    args = parser.parse_args()

    write_readme(args.model_path)
    upload_args = {"model_path": args.model_path, "repo_id": args.repo_id}
    if args.token:
        upload_args["token"] = args.token
    upload_to_repo(**upload_args)

if __name__ == "__main__":
    main()