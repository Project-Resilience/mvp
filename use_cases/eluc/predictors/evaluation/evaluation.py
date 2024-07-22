"""
Class to evaluate predictors given a config on a dataset.
Also a script to demo how it works.
"""
import importlib
import json
from pathlib import Path

import pandas as pd

import data.constants as constants
from data.eluc_data import ELUCData
from persistence.persistors.hf_persistor import HuggingFacePersistor
from predictors.predictor import Predictor
from predictors.evaluation.validator import Validator

class Evaluator():
    """
    Evaluator class to evaluate predictors on a dataset.
    Uses a config to dynamically load predictors.
    The config must point to the classpath of a serializer that can call .load() to return a Predictor object.
    Alternatively, it may use a HuggingFace url to download a model to a given path, THEN load with the serializer.
    """
    def __init__(self, config: dict):
        """
        Initializes the Evaluator with the custom classes it has to load.
        """
        self.predictors = self.dynamically_load_models(config)
        # We don't pass change into the outcomes column.
        self.validator = Validator(constants.CAO_MAPPING["context"], constants.CAO_MAPPING["actions"], ["ELUC"])

    def dynamically_load_models(self, config: dict) -> list[Predictor]:
        """
        Uses importlib to dynamically load models from a config.
        Config must have a list of models with the following:
            - type: "hf" or "local" to determine if it is a HuggingFace model or local model.
            - name: name of the serializer class to load.
            - classpath: path to the class that calls .load()
            - filepath: path to the model on disk or where to save the HuggingFace model.
            - (optional) url: url to download the model from HuggingFace.
        Returns a dict with keys being the filepath and values being the Predictor object.
        """
        predictors = {}
        for model in config["models"]:
            # We dynamically instantiate model_instance as some sort of class that can handle .load() and returns
            # a Predictor object.
            spec = importlib.util.spec_from_file_location(model["name"], model["classpath"])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model_instance = getattr(module, model["name"])

            # Once we have our model_instance we can load the model from disk or from HuggingFace.
            if model["type"] == "hf":
                persistor = HuggingFacePersistor(model_instance())
                predictor = persistor.from_pretrained(model["url"], local_dir=model["filepath"])
            elif model["type"] == "local":
                predictor = model_instance().load(Path(model["filepath"]))
            else:
                raise ValueError("Model type must be either 'hf' or 'local'")
            predictors[model["filepath"]] = predictor
        return predictors
    
    def evaluate(self, test_df: pd.DataFrame):
        """
        Evaluates our list of predictors on a given test dataframe.
        The dataframe is expected to be raw data.
        """
        y_true = test_df["ELUC"]
        test_df = self.validator.validate_input(test_df)
        results = {}
        for predictor_path, predictor in self.predictors.items():
            outcome_df = predictor.predict(test_df)
            assert self.validator.validate_output(test_df, outcome_df)
            y_pred = outcome_df["ELUC"]
            mae = (y_true - y_pred).abs().mean()
            results[predictor_path] = mae
        return results

def run_evaluation():
    """
    A demo script to show how the Evaluator class works.
    """
    print("Evaluating models in config.json...")
    config = json.load(open(Path("predictors/evaluation/config.json"), "r", encoding="utf-8"))
    evaluator = Evaluator(config)
    dataset = ELUCData.from_hf()
    results = evaluator.evaluate(dataset.test_df)
    print("Results:")
    print(results)

if __name__ == "__main__":
    run_evaluation()
