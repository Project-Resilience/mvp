# MVP Climate Change Demo

This is a demo of the MVP Climate Change app. It allows users to select a location and year from a map of the UK, Switzerland, or Brazil, see its land use composition, and prescribe or manually make changes to it and see the predicted ELUC (Emissions from Land Use Change). It is a simple Dash app.

## Dependencies:

The demo relies on the `unileaf_util` package. This has to be manually installed from the `.whl` file.
The requirements.txt is set up for M1 macs. Your installations of tensorflow and pytorch may differ.

## Running the app:

To run the app use: ``python app.py``

## Predictors:

Saved predictors can be found in `predictors/`. The XGBoost predictor's weights are stored in a `.json` file whereas the LSTM predictor's weights are stored in a `.pt` file and its configuration is saved in the corresponding `.json` file.

## Prescriptors:

Prescriptors are stored in `prescriptors/` as well as the pareto front image and a CSV of pareto info from training the prescriptors.