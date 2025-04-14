# Emissions from Land Use Change (ELUC)

Goal: provide decision makers with tools to know how their land-use choices affect CO2 fluxes in the long-term.  

The tools should help decisions makers with their choices:
- For a geographical grid cell, identified by its latitude and longitude, what changes to the land usage can be
made to reduce CO2 emissions?
- What will be the long term CO2 impact of changing the land usage in a particular way?

It is possible to learn from historical decisions made by decision makers all around the world if they can be compared.

Work from this project was published in [NeurIPS 2023 Workshop: Tackling Climate Change with Machine Learning](https://www.climatechange.ai/events/neurips2023) as a paper: [Discovering Effective Policies for Land-Use Planning](https://nn.cs.utexas.edu/?miikkulainen:arxiv23) which won the *Best Pathway to Impact* award. The recorded talk can be found [here](https://www.climatechange.ai/papers/neurips2023/94). See the Experiments section for details on how to replicate the results in the paper.

## Data

### Download

A dataset consisting of land-use changes and their committed emissions can be found on [HuggingFace](https://huggingface.co/datasets/projectresilience/ELUC-committed). The raw data used to generate this dataset can also be found within the HuggingFace repo. Further details about the dataset can be found below.

### ELUC

BLUE simulations with committed emissions could be used to estimate the long-term CO2 impact.
"Committed emissions" means all the emissions that are caused by a land-use change event are attributed to the year
of the event.
BLUE (bookkeeping of land use emissions) is a bookkeeping model that attributes carbon fluxes to land use activities.
See [BLUE: Bookkeeping of Land Use Emissions](https://doi.org/10.1002/2014GB004997) for more details.

### LUC

The Land Use Change (LUC) data is provided by the LUH2 project [(LUH2: Land Use Harmonization 2)](https://luh.umd.edu/)
The land-use harmonization strategy estimates the fractional land-use patterns, underlying land-use transitions,
and key agricultural management information, annually for the time period 850-2100 at 0.25 x 0.25 resolution.  

The LUH2 model and datasets prepared for CMIP6 are described in
[Hurtt et al. 2020.](https://gmd.copernicus.org/articles/13/5425/2020/gmd-13-5425-2020-discussion.html)

Land-use Harmonization (LUH) data for [GCB 2022](https://doi.org/10.5194/essd-14-4811-2022) is provided in
3 separate files, which can be downloaded from the following links
(for the states, transitions, and management data layers respectively):

http://luh.umd.edu/LUH2/LUH2_GCB_2022/states.nc  
http://luh.umd.edu/LUH2/LUH2_GCB_2022/transitions.nc  
http://luh.umd.edu/LUH2/LUH2_GCB_2022/management.nc  

These files are based on the new HYDE3.3, as well as the 2021 FAO wood harvest data, for all years 850-2022.

The data files are for the years 850-2022, which keeps the file format consistent with the LUH2 data produced for CMIP6,
hence the start year of 850. The LUH2-GCB2022 data will be different from the LUH2 v2h data used for CMIP6 for all
years, due to the use of the new HYDE3.3 crop/grazing land dataset.

See https://luh.umd.edu/  for more details.

### Land Use Types

Primary: Vegetation that is untouched by humans
- **primf**: Primary forest
- **primn**: Primary nonforest vegetation

Secondary: Vegetation that has been touched by humans
- **secdf**: Secondary forest
- **secdn**: Secondary nonforest vegetation

Urban
- **urban**: Urban areas

Crop
- **c3ann**: Annual C3 crops (e.g. wheat)
- **c4ann**: Annual C4 crops (e.g. maize)
- **c3per**: Perennial C3 crops (e.g. banana)
- **c4per**: Perennial C4 crops (e.g. sugarcane)
- **c3nfx**: Nitrogen fixing C3 crops (e.g. soybean)

Pasture
- **pastr**: Managed pasture land
- **range**: Natural grassland / savannah / desert / etc.

## Modeling decisions

A decision can be represented by 3 constituents: **context**, **actions** and **outcomes**

> In a particular **context**, what were the **outcomes** of the decision maker's **actions**?

In other words, in a particular situation, what were the results of the decisions of the decision maker?

### Context

The context describes the current situation the decision maker is in.
For the land use change use case this is a particular grid cell,
a point in time when the decision had to be made, and the current usage of the land:

- Latitude
- Longitude
- Area
- Year
- Land usage, as a percentage (does not necessarily sum to 100%)
  - primf
  - primn
  - sedf
  - secdn
  - urban
  - c3ann
  - c4ann
  - c3per
  - c4per
  - c3fnx
  - pastr
  - range

Latitude and longitude represent the cell on the grid.  
Area represents the surface of the cell. Cells close to the equator have a bigger area than cells close to the poles.  
Year is useful to capture historical decisions: the same cell has been through a lot of land use changes
over the years.  
Land usage represents the percentage of the land used by each land type. Note that the land usage does not sum to 100% because of area in the cell that is not land (e.g typically sea, lake, etc.).

### Actions

Actions represent the decisions decision makers can make. How can they change the land?
We considered 2 limitations:
1. Decision makers can't affect primary land:
   1. It's better to preserve primary vegetation. Destroying it is not an option.
   2. It's not possible to re-plant primary vegetation. Once destroyed, it's destroyed forever and
   can't be planted back (it would become secondary vegeation)
2. Decision makers can't affect urban areas. We consider the needs for larger / smaller urban areas are dictated by
   other imperatives, other decision makers.
   1. It doesn't seem reasonable to recommend to destroy a city
   2. It doesn't seem reasonable to recommend to expand a city 

### Outcomes

- Committed Emissions from Land Use Change (ELUC): all present and future CO2 emissions, in metric ton of carbon per hectare (tC/ha),
resulting from the land use change
- Percentage of land that was changed

There is a trade-off between these 2 objectives: it easy to reduce emissions by changing
most of the land, but that would come at a huge cost. This "cost" can be approximately derived from
the percentage of land that was changed. In other words decision makers have to:
- minimize ELUC
- while minimizing land change at the same time

## Modeling

*Note: Before running anything make sure to set your python path environment variable with: `export PYTHONPATH=$PWD` while in the eluc directory.*

Additionaly, the Project Resilience SDK is required to use the MVP. Currently it is not hosted on PyPi so it must be installed manually. To do so, clone the [Project Resilience SDK](https://github.com/Project-Resilience/sdk) and run pip install . in its root directory.

### Predictions

Given **context** and **actions** -> predict **outcomes**  
Given the land usage of a specific location, and the changes that were made during a specific year,
predict the CO2 long term emissions directly caused by these changes (ELUC).

This is a prediction problem.

Anyone can contribute a prediction model, as long as it complies with the `predictor` interface
and its inputs and outputs.

Code to train predictor models can be found in the [predictors](predictors) directory. Currently, the following models are implemented:
- [Neural Network](predictors/neural_network/neural_net_predictor.py)
- [Random Forest](predictors/sklearn/sklearn_predictor.py)
- [Linear Regression](predictors/sklearn/sklearn_predictor.py)

### Ensembling

Ensemble models can be constructed from predicition models.

TODO: see task #53

### Uncertainty

Evaluate point-prediction uncertainty

TODO: see task #55

### Prescriptions

Given **context** -> prescribe **actions** that optimize **outcomes**

This is an optimization problem.
Anyone can contribute a prescription model, as long as it complies with the `prescriptor` interface
and its inputs and outputs.

Code to train prescriptor models can be found in the [prescriptors](prescriptors) directory. Currently 2 prescriptors are implemented:
- [UniLEAF Prescriptor](prescriptors/esp/unileaf_prescriptor.py) (note: this prescriptor uses the ESP SDK which is not open source, however it remains in the repo as it was used in the original paper)
- [Torch Prescriptor](prescriptors/nsga2/torch_prescriptor.py) which is an open-source method that implements the NSGA-II algorithm in PyTorch.

To train a prescriptor, use the [train_prescriptors.py](prescriptors/nsga2/train_prescriptors.py) script. A predictor model needs to be trained beforehand. A template config to be used in prescriptor training can be found in the [configs](prescriptors/nsga2/configs/test.json) folder. Seeds can also be trained to allow the prescriptor to find candidates along the edge of the pareto-front. These can be trained using [train_seeds.py](prescriptors/nsga2/train_seeds.py).

## Experiments
Experiments run to analyze the models for the paper can be found in the [experiments](experiments) directory. Rough [predictor](experiments/predictor_experiments.ipynb) and [prescriptor](experiments/prescriptor_experiments.ipynb) experiments can be found as well as more polished notebooks such as [crop.ipynb](experiments/crop.ipynb) which are used to generate figures for the paper.

Ultimately, to overall replicate the paper, the following steps should be taken:
1. Train the predictor models using [train_predictors.py](predictors/train_predictors.py)

2. Run predictor significance using [predictor_experiments.ipynb](experiments/predictor_experiments.ipynb)

3. Train seed prescriptor models with [train_seeds.py](prescriptors/nsga2/train_seeds.py)

4. Run evolution to train prescriptor models with [train_prescriptors.py](prescriptors/nsga2/train_prescriptors.py)

5. Run prescriptor analysis using notebooks in [experiments](experiments) such as [crop.ipynb](experiments/crop.ipynb)

## Robojudge

"Robojudge" is an interactive tool that can be used to compare predictors.
Some models perform better depending on the evaluation metric, the countries or the years on which they are evaluated.

Preliminary evaluation code for predictors can be found in the [demo_predictors.ipynb](predictors/demo_predictors.ipynb) notebook.

TODO: see task #49

## Demo

A user interface for decision makers is available here: https://landuse.evolution.ml/

To run the app, first download the preprocessed dataset from [HuggingFace](https://huggingface.co/datasets/projectresilience/land-use-app-data) using ```python -m app.process_data```.

Then run the app using ```python -m app.app```.

In order to build and run a docker image containing the app, first build with 

```docker build -t landusedemo .```

and then run with

```docker run -p 8080:4057 --name landuse-demo-container landusedemo```.

## Testing

To run unit tests, use the following command: ```python -m unittest```.

TODO: see task #79

To run pylint, use the following command: ```pylint .```

## References

ELUC data provided by the BLUE model [(BLUE: Bookkeeping of land use emissions)](https://doi.org/10.1002/2014GB004997)

Land use change data provided by the LUH2 project [(LUH2: Land Use Harmonization 2)](https://luh.umd.edu/)

Setup is described in Appendix C2.1 of the GCB 2022 report [(Global Carbon Budget 2022 report)](https://doi.org/10.5194/essd-14-4811-2022)

The Global Carbon Budget report assesses the global CO2 budget for the Intergovernmental Panel on Climate Change [(IPCC)](https://www.ipcc.ch/)
