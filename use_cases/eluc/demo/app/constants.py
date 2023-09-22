import base64
import os
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_PATH = os.path.join(ROOT_DIR, "../data/processed/eluc_1982.csv")

GRID_STEP = 0.25

INDEX_COLS = ["time", "lat", "lon"]

LAND_USE_COLS = ['c3ann', 'c3nfx', 'c3per', 'c4ann', 'c4per', 'pastr', 'primf', 'primn', 'range', 'secdf', 'secdn', 'urban']
CONTEXT_COLUMNS = LAND_USE_COLS + ['cell_area']
DIFF_LAND_USE_COLS = [f"{col}_diff" for col in LAND_USE_COLS]
COLS_MAP = dict(zip(LAND_USE_COLS, DIFF_LAND_USE_COLS))

# Prescriptor outputs
RECO_COLS = ['c3ann', 'c3nfx', 'c3per','c4ann', 'c4per', 'pastr', 'range', 'secdf', 'secdn']
DIFF_RECO_COLS = [f"{col}_diff" for col in RECO_COLS]
RECO_MAP = dict(zip(RECO_COLS, DIFF_RECO_COLS))

NO_CHANGE_COLS = ["primf", "primn", "urban"]
CHART_COLS = LAND_USE_COLS + ["nonland"]

SLIDER_PRECISION = 1e-5

# Tonnes of CO2 per person for a flight from JFK to Geneva
CO2_JFK_GVA = 2.2
CO2_PERSON = 4

# For creating treemap
C3 = ['c3ann', 'c3nfx', 'c3per']
C4 = ['c4ann', 'c4per']
PRIMARY = ['primf', 'primn']
SECONDARY = ['secdf', 'secdn']
FIELDS = ['pastr', 'range']

CHART_TYPES = ["Treemap", "Pie Chart"]

PREDICTOR_PATH = os.path.join(ROOT_DIR, "../predictors/")
PRESCRIPTOR_PATH = os.path.join(ROOT_DIR, "../prescriptors/")

# Pareto front
PARETO_CSV_PATH = os.path.join(PRESCRIPTOR_PATH, "pareto.csv")
PARETO_FRONT_PATH = os.path.join(PRESCRIPTOR_PATH, "pareto_front.png")
PARETO_FRONT = base64.b64encode(open(PARETO_FRONT_PATH, 'rb').read()).decode('ascii')

FIELDS_PATH = os.path.join(PRESCRIPTOR_PATH, "fields.json")

DEFAULT_PRESCRIPTOR_IDX = 3  # By default we select the fourth prescriptor that minimizes change
