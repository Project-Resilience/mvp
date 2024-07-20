"""
Class to evaluate predictors given a config on a dataset.
Also a script to demo how it works.
"""
import importlib
from pathlib import Path

import pandas as pd

from data.eluc_data import ELUCData
from persistence.persistors.hf_persistor import HuggingFacePersistor
from predictors.predictor import Predictor

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
        results = {}
        y_true = test_df["ELUC"]
        for predictor_path, predictor in self.predictors.items():
            outcome_df = predictor.predict(test_df)
            y_pred = outcome_df["ELUC"]
            mae = (y_true - y_pred).abs().mean()
            results[predictor_path] = mae
        return results

def main():
    """
    A demo script to show how the Evaluator class works.
    """
    config = {
        "models": [
            {
                "type": "local",
                "name": "TemplatePredictor",
                "classpath": "predictors/custom/template/template_predictor.py",
                "filepath": "predictors/custom/template/model.pt"
            },
            {
                "type": "hf",
                "name": "NeuralNetSerializer",
                "classpath": "persistence/serializers/neural_network_serializer.py",
                "url": "danyoung/eluc-global-nn",
                "filepath": "predictors/trained_models/danyoung--eluc-global-nn"
            },
            {
                "type": "hf",
                "name": "SKLearnSerializer",
                "url": "danyoung/eluc-global-linreg",
                "classpath": "persistence/serializers/sklearn_serializer.py",
                "filepath": "predictors/trained_models/danyoung--eluc-global-linreg"
            },
            {
                "type": "hf",
                "name": "SKLearnSerializer",
                "url": "danyoung/eluc-global-rf",
                "classpath": "persistence/serializers/sklearn_serializer.py",
                "filepath": "predictors/trained_models/danyoung--eluc-global-rf"
            }
        ]
    }
    evaluator = Evaluator(config)
    dataset = ELUCData.from_hf()
    results = evaluator.evaluate(dataset.test_df)
    print(results)

if __name__ == "__main__":
    main()
