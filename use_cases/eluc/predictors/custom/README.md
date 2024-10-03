# Custom Predictors

This directory contains custom predictors that can be used with the ELUC use case. Since percent change is measurable, we look for predictors that can predict ELUC.

## Create a Custom Predictor

An example custom predictor can be found in the [template](template) folder. In order to create a custom predictor, 2 steps must be completed.

1. You need to implement the `Predictor` interface. This is defined in [predictor.py](../predictor.py). It is a simple abstract class that requires a `predict` method that takes in a dataframe of context and actions and returns a dataframe of outcomes.

2. You need, either in the same class or a specific serializer class, to implement a `load` method that takes in a path to a model on disk and returns an instance of the `Predictor`. (See [serializer.py](../../persistence/serializers/serializer.py) for the interface for serialization and [neural_network_serializer.py](../../persistence/serializers/neural_network_serializer.py) for an example of how to implement serialization.)

Finally, you must add your custom predictor to the [config](../scoring/config.json) file in order to score it.

### Load from HuggingFace

To load a custom model saved on HuggingFace, see the [HuggingFacePersistor](../../persistence/persistors/hf_persistor.py) class. It takes in a `FileSerializer` to download a HuggingFace model to disk then load it. An example of how to score a model from HuggingFace can be found in the [config](../scoring/config.json).