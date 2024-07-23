"""
Unit tests for the predictors.
"""
import unittest
import shutil
from pathlib import Path

import pandas as pd

from persistence.serializers.neural_network_serializer import NeuralNetSerializer
from persistence.serializers.sklearn_serializer import SKLearnSerializer
from predictors.neural_network.neural_net_predictor import NeuralNetPredictor
from predictors.sklearn_predictor.sklearn_predictor import LinearRegressionPredictor, RandomForestPredictor

class TestPredictors(unittest.TestCase):
    """
    Tests the 3 base predictor implementations' saving and loading behavior.
    """
    def setUp(self):
        """
        We set the models up like this so that in test_loaded_same we can instantiate
        2 models with the same parameters, load one from the other's save, and check if
        their predictions are the same.
        """
        self.models = [
            NeuralNetPredictor,
            LinearRegressionPredictor,
            RandomForestPredictor
        ]
        self.serializers = [
            NeuralNetSerializer(),
            SKLearnSerializer(),
            SKLearnSerializer()
        ]
        self.configs = [
            {'hidden_sizes': [4], 'epochs': 1, 'batch_size': 1, 'device': 'cpu'},
            {'n_jobs': -1},
            {'n_jobs': -1, "n_estimators": 10, "max_depth": 2}
        ]
        self.dummy_data = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 5, 6, 4], "c": [7, 8, 9, 4]})
        self.dummy_target = pd.Series([1, 2, 3, 4], name="label")
        self.temp_path = Path("tests/temp")

    def test_save_file_names(self):
        """
        Checks to make sure the model's save method creates the correct files.
        """
        save_file_names = [
            ["model.pt", "config.json", "scaler.joblib"],
            ["model.joblib", "config.json"],
            ["model.joblib", "config.json"]
        ]
        for model, serializer, config, test_names in zip(self.models, self.serializers, self.configs, save_file_names):
            with self.subTest(model=model):
                predictor = model(config)
                predictor.fit(self.dummy_data, self.dummy_target)
                serializer.save(predictor, self.temp_path)
                files = [f.name for f in self.temp_path.glob("**/*") if f.is_file()]
                self.assertEqual(set(files), set(test_names))
                shutil.rmtree(self.temp_path)
                self.assertFalse(self.temp_path.exists())

    def test_loaded_same(self):
        """
        Makes sure a predictor's predictions are consistent before and after saving/loading.
        Fits a predictor then saves and loads it, then checks if the predictions are the same.
        """

        for model, serializer, config in zip(self.models, self.serializers, self.configs):
            with self.subTest(model=model):
                predictor = model(config)
                predictor.fit(self.dummy_data.iloc[:2], self.dummy_target.iloc[:2])
                output = predictor.predict(self.dummy_data.iloc[2:])
                serializer.save(predictor, self.temp_path)

                loaded = serializer.load(self.temp_path)
                loaded_output = loaded.predict(self.dummy_data.iloc[2:])

                self.assertTrue((output == loaded_output).all().all()) # Pandas is so annoying why is this necessary?
                shutil.rmtree(self.temp_path)
                self.assertFalse(self.temp_path.exists())

    def tearDown(self):
        """
        Removes the temp directory if it exists.
        """
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)


class TestNeuralNet(unittest.TestCase):
    """
    Specifically tests the neural net predictor
    """

    def test_single_input(self):
        """
        Tests the neural net with a single input.
        """
        predictor = NeuralNetPredictor({"hidden_sizes": [4], "epochs": 1, "batch_size": 1, "device": "cpu"})

        train_data = pd.DataFrame({"a": [1], "b": [2], "c": [3], "label": [4]})
        test_data = pd.DataFrame({"a": [4], "b": [5], "c": [6]})

        predictor.fit(train_data[['a', 'b', 'c']], train_data['label'])
        out = predictor.predict(test_data)
        self.assertEqual(out.shape, (1, 1))

    def test_multi_input(self):
        """
        Tests the neural net with multiple inputs.
        """
        predictor = NeuralNetPredictor({"hidden_sizes": [4], "epochs": 1, "batch_size": 1, "device": "cpu"})

        train_data = pd.DataFrame({"a": [1, 2], "b": [2, 3], "c": [3, 4], "label": [4, 5]})
        test_data = pd.DataFrame({"a": [4, 5], "b": [5, 6], "c": [6, 7]})

        predictor.fit(train_data[['a', 'b', 'c']], train_data['label'])
        out = predictor.predict(test_data)
        self.assertEqual(out.shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
