import os
from pathlib import Path

from data.constants import LAND_USE_COLS

DATA_FILE_PATH = Path("data/processed/app_data.csv")

APP_START_YEAR = 2012

GRID_STEP = 0.25

INDEX_COLS = ["time", "lat", "lon"]

NO_CHANGE_COLS = ["primf", "primn", "urban"]
CHART_COLS = LAND_USE_COLS + ["nonland"]

SLIDER_PRECISION = 1e-5

# Tonnes of CO2 per person for a flight from JFK to Geneva
CO2_JFK_GVA = 2.2
CO2_PERSON = 4

# For creating treemap
PRIMARY = ['primf', 'primn']
SECONDARY = ['secdf', 'secdn']
FIELDS = ['pastr', 'range']

CHART_TYPES = ["Treemap", "Pie Chart"]

PREDICTOR_PATH = Path("predictors/trained_models")
PRESCRIPTOR_PATH = Path("prescriptors/nsga2/app_prescriptors")

# Pareto front
# PARETO_CSV_PATH = os.path.join(PRESCRIPTOR_PATH, "pareto.csv")
# PARETO_FRONT_PATH = os.path.join(PRESCRIPTOR_PATH, "pareto_front.png")
# PARETO_FRONT = base64.b64encode(open(PARETO_FRONT_PATH, 'rb').read()).decode('ascii')

FIELDS_PATH = PRESCRIPTOR_PATH / "fields.json"

DEFAULT_PRESCRIPTOR_IDX = 3  # By default we select the fourth prescriptor that minimizes change
