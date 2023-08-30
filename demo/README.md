# MVP Climate Change Demo

This is a demo of the MVP Climate Change app. It allows users to select a location and year from a map of the UK, Switzerland, or Brazil, see its land use composition, and prescribe or manually make changes to it and see the predicted ELUC (Emissions from Land Use Change). It is a simple Dash app.

## Dependencies:

This application relies on the ``unileaf-util`` package which can be downloaded from git. Save your authentication token in ``$LEAF_PRIVATE_SOURCE_CREDENTIAL`` in order to be able to ``pip install -r requirements.txt`` for the dependencies.

## Downloading the data:

In ``data/`` there is a script called ``process_data.py``. This will download the entire 2.5GB data file from HuggingFace then process it into a 500MB csv that is used by the app. A token is required to download the data and must be saved in ``$HF_TOKEN``.

## Predictors:

The RandomForest model is 1.7GB and is also saved on HuggingFace. To download it run ``download_predictors.py`` in ``predictors/``. This downloads a ``.joblib`` file that is loaded in the app.

## Prescriptors:

Prescriptors are already stored in `prescriptors/` as well as the pareto front image and a CSV of pareto info from training the prescriptors.

## Testing:

Testing can be done with ``python -m unittest discover``

## Running the app:

To run the app call the app module with ``python -m app.app`` or use gunicorn with ``gunicorn -b 0.0.0.0:4057 app.app:server``.

## Deployment:

Once ``process_data.py`` and ``download_predictors.py`` have been run, the app can be deployed by building with:
```
docker build \
--build-arg leaf_private_source_credentials=$LEAF_PRIVATE_SOURCE_CREDENTIALS \
-t eluc-demo .
```
then the container can be run with:
```
docker run \
-v PATH_TO_PROJECT/mvp/demo/data/processed:/usr/local/cognizant/eluc/data/processed:ro \
-v PATH_TO_PROJECT/mvp/demo/predictors:/usr/local/cognizant/eluc/predictors:ro \
-p 8080:4057 eluc-demo
```
Note: This mounts your local directories to the docker container, different steps may have to be taken for different setups.