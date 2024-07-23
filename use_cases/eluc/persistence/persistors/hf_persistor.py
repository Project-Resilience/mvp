"""
Persistor for models to and from HuggingFace repo.
"""
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from persistence.persistors.persistor import Persistor


class HuggingFacePersistor(Persistor):
    """
    Persists models to and from HuggingFace repo.
    """
    def write_readme(self, model_path: str):
        """
        Writes readme to model save path to upload.
        TODO: Need to add more info to the readme and make it a proper template.
        """
        model_path = Path(model_path)
        with open(model_path / "README.md", "w", encoding="utf-8") as file:
            file.write("This is a demo model created for project resilience")

    def persist(self, model, model_path: Path, repo_id: str, **persistence_args):
        """
        Serializes the model to a local path using the file_serializer,
        then uploads the model to a HuggingFace repo.
        """
        # Save model and write readme
        self.file_serializer.save(model, model_path)
        self.write_readme(model_path)

        # Get token if it exists
        token = persistence_args.get("token", None)

        api = HfApi()
        # Create repo if it doesn't exist
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            token=token
        )

        # Upload model to repo
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            token=token
        )

    def from_pretrained(self, path_or_url: str, **hf_args):
        """
        Loads a model from a path or if it is not found, from a huggingface repo.
        :param path_or_url: path to the model or url to the huggingface repo.
        :param hf_args: arguments to pass to the snapshot_download function from huggingface.
        """
        path = Path(path_or_url)
        if path.exists() and path.is_dir():
            return self.file_serializer.load(path)
        # TODO: Need a try except block to catch download errors
        url_path = path_or_url.replace("/", "--")
        local_dir = hf_args.get("local_dir", f"huggingface_models/{url_path}")

        if not Path(local_dir).exists() or not Path(local_dir).is_dir():
            hf_args["local_dir"] = local_dir
            snapshot_download(repo_id=path_or_url, **hf_args)

        return self.file_serializer.load(Path(local_dir))
