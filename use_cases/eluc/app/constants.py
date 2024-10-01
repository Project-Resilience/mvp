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
