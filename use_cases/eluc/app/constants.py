"""
Constants for the demo application.
"""
from pathlib import Path

from data.constants import LAND_USE_COLS

DATA_FILE_PATH = Path("app/data/app_data.csv")

APP_START_YEAR = 2012

GRID_STEP = 0.25

INDEX_COLS = ["time_idx", "lat_idx", "lon_idx"]

NO_CHANGE_COLS = ["primf", "primn", "urban"]
CHART_COLS = LAND_USE_COLS + ["nonland"]

SLIDER_PRECISION = 1e-5

# Tonnes of CO2 per person for a flight from JFK to Geneva
TC_TO_TCO2 = 3.67
CO2_JFK_GVA = 2.2
CO2_PERSON = 4

# For creating treemap
PRIMARY = ['primf', 'primn']
SECONDARY = ['secdf', 'secdn']
FIELDS = ['pastr', 'range']

CHART_TYPES = ["Treemap", "Pie Chart"]

PREDICTOR_PATH = Path("predictors/trained_models")
PRESCRIPTOR_PATH = Path("prescriptors/trained_models")

# Pareto front
PARETO_CSV_PATH = Path("app/data/pareto.csv")

DEFAULT_PRESCRIPTOR_IDX = 1  # By default we select the second prescriptor that minimizes change

DESC_TEXT = "mb-5 w-75 text-center"

JUMBOTRON = "p-3 bg-white rounded-5 mx-auto w-75 mb-3"

CONTAINER = "py-3 d-flex flex-column h-100 align-items-center"

HEADER = "text-center mb-5"

