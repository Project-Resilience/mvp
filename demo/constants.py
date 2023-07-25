import base64

# TODO: Have this generated somewhere else/loaded from a file
fields = {
    'ELUC': {'data_type': 'FLOAT', 'has_nan': False, 'mean': -0.11856405, 'range': [-38.421227, 22.764368], 'std_dev': 1.0573674, 'sum': -54801.844, 'valued': 'CONTINUOUS'}, 
    'c3ann': {'data_type': 'FLOAT', 'has_nan': False, 'mean': 0.26532868, 'range': [0, 0.9289606], 'std_dev': 0.16872787, 'sum': 122638.36, 'valued': 'CONTINUOUS'}, 
    'c3nfx': {'data_type': 'FLOAT', 'has_nan': False, 'mean': 0.015724484, 'range': [0, 0.09423922], 'std_dev': 0.013063688, 'sum': 7268.061, 'valued': 'CONTINUOUS'},
    'c3per': {'data_type': 'FLOAT', 'has_nan': False, 'mean': 0.013562751, 'range': [0, 0.37380314], 'std_dev': 0.030480456, 'sum': 6268.88, 'valued': 'CONTINUOUS'},
    'c4ann': {'data_type': 'FLOAT', 'has_nan': False, 'mean': 0.047797143, 'range': [0, 0.44018674], 'std_dev': 0.04781523, 'sum': 22092.46, 'valued': 'CONTINUOUS'},
    'c4per': {'data_type': 'FLOAT', 'has_nan': False, 'mean': 1.8471316e-09, 'range': [0, 2.1909951e-05], 'std_dev': 9.993674e-08, 'sum': 0.00085376826, 'valued': 'CONTINUOUS'},
    'pastr': {'data_type': 'FLOAT', 'has_nan': False, 'mean': 0.2755411, 'range': [0, 0.9999977], 'std_dev': 0.18394634, 'sum': 127358.67, 'valued': 'CONTINUOUS'},
    'primf': {'data_type': 'FLOAT', 'has_nan': False, 'mean': 0.0408774, 'range': [0, 1], 'std_dev': 0.19624169, 'sum': 18894.066, 'valued': 'CONTINUOUS'},
    'primn': {'data_type': 'FLOAT', 'has_nan': False, 'mean': 0.0010933084, 'range': [0, 0.81347555], 'std_dev': 0.023737159, 'sum': 505.3414, 'valued': 'CONTINUOUS'},
    'range': {'data_type': 'FLOAT', 'has_nan': False, 'mean': 0.015515061, 'range': [0, 0.76993644], 'std_dev': 0.064742275, 'sum': 7171.2627, 'valued': 'CONTINUOUS'},
    'secdf': {'data_type': 'FLOAT', 'has_nan': False, 'mean': 0.12541063, 'range': [0, 1], 'std_dev': 0.2203229, 'sum': 57966.426, 'valued': 'CONTINUOUS'},
    'secdn': {'data_type': 'FLOAT', 'has_nan': False, 'mean': 0.14883429, 'range': [0, 0.9977025], 'std_dev': 0.2135971, 'sum': 68793.15, 'valued': 'CONTINUOUS'},
    'urban': {'data_type': 'FLOAT', 'has_nan': False, 'mean': 0.023921186, 'range': [0, 0.9804862], 'std_dev': 0.0599659, 'sum': 11056.683, 'valued': 'CONTINUOUS'},
    'change': {'data_type': 'FLOAT', 'has_nan': False, 'mean': 0.004188048, 'range': [0, 0.3609924], 'std_dev': 0.009503605, 'sum': 1935.7703, 'valued': 'CONTINUOUS'},
    'cell_area': {'data_type': 'FLOAT', 'has_nan': False, 'mean': 50892.78, 'range': [40233.22, 77223.74], 'std_dev': 6527.7935, 'sum': 23523305000, 'valued': 'CONTINUOUS'},
    'c3ann_diff': {'data_type': 'FLOAT', 'has_nan': False, 'mean': -0.00044108444, 'range': [-0.28561932, 0.09917167], 'std_dev': 0.0037445952, 'sum': -203.87495, 'valued': 'CONTINUOUS'},
    'c3nfx_diff': {'data_type': 'FLOAT', 'has_nan': False, 'mean': -2.9746114e-05, 'range': [-0.015432712, 0.014108786], 'std_dev': 0.00022670913, 'sum': -13.749041, 'valued': 'CONTINUOUS'},
    'c3per_diff': {'data_type': 'FLOAT', 'has_nan': False, 'mean': -2.4286626e-05, 'range': [-0.02339302, 0.022070948], 'std_dev': 0.0003331531, 'sum': -11.2255945, 'valued': 'CONTINUOUS'},
    'c4ann_diff': {'data_type': 'FLOAT', 'has_nan': False, 'mean': -7.317337e-05, 'range': [-0.054494824, 0.05511633], 'std_dev': 0.00080506655, 'sum': -33.821682, 'valued': 'CONTINUOUS'},
    'c4per_diff': {'data_type': 'FLOAT', 'has_nan': False, 'mean': 3.2219023e-11, 'range': [-1.3854105e-06, 8.639275e-06], 'std_dev': 1.55411e-08, 'sum': 1.4892052e-05, 'valued': 'CONTINUOUS'},
    'pastr_diff': {'data_type': 'FLOAT', 'has_nan': False, 'mean': -0.00056610367, 'range': [-0.3609924, 0.13539696], 'std_dev': 0.008346922, 'sum': -261.6605, 'valued': 'CONTINUOUS'},
    'primf_diff': {'data_type': 'FLOAT', 'has_nan': False, 'mean': -9.644197e-06, 'range': [-0.014856517, 0], 'std_dev': 0.00010567804, 'sum': -4.4576735, 'valued': 'CONTINUOUS'},
    'primn_diff': {'data_type': 'FLOAT', 'has_nan': False, 'mean': -1.4477387e-06, 'range': [-0.046862364, 0], 'std_dev': 9.3433555e-05, 'sum': -0.66916364, 'valued': 'CONTINUOUS'},
    'range_diff': {'data_type': 'FLOAT', 'has_nan': False, 'mean': 5.9423666e-05, 'range': [-0.16614735, 0.16141176], 'std_dev': 0.0032823936, 'sum': 27.46639, 'valued': 'CONTINUOUS'},
    'secdf_diff': {'data_type': 'FLOAT', 'has_nan': False, 'mean': 0.00023636929, 'range': [-0.17319721, 0.35972273], 'std_dev': 0.0065409928, 'sum': 109.25296, 'valued': 'CONTINUOUS'},
    'secdn_diff': {'data_type': 'FLOAT', 'has_nan': False, 'mean': 0.0006134256, 'range': [-0.1592417, 0.3085313], 'std_dev': 0.0069925156, 'sum': 283.5333, 'valued': 'CONTINUOUS'},
    'urban_diff': {'data_type': 'FLOAT', 'has_nan': False, 'mean': 0.00023626754, 'range': [-0.034920335, 0.11945325], 'std_dev': 0.000924528, 'sum': 109.20593, 'valued': 'CONTINUOUS'}}

cao_mapping = {
    'context': ['lat', 'lon', 'time', 'c3ann', 'c3nfx', 'c3per', 'c4ann', 'i_lat', 'i_lon', 'pastr', 'primf', 'primn', 'range', 'secdf', 'secdn', 'urban', 'cell_area'],
    'actions': ['c3ann_diff', 'c3nfx_diff', 'c3per_diff', 'c4ann_diff', 'pastr_diff', 'range_diff', 'secdf_diff', 'secdn_diff', 'urban_diff'],
    'outcomes': ['ELUC', 'Change']}

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

DEFAULT_PRESCRIPTOR_IDX = 1  # By default we select the second prescriptor that minimizes change
PREDICTOR_LIST = ["Random Forest"]

SLIDER_PRECISION = 1e-5

MAP_COORDINATE_DICT = {
    "UK": {"lat": 54.5, "lon": -2.5, "zoom": 20}, 
    "Brazil": {"lat": -12, "lon": -51.2, "zoom": 4},
    "Switzerland": {"lat": 47, "lon": 8.15, "zoom": 20}
}

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

# Pareto front
PARETO_CSV_PATH = "prescriptors/pareto.csv"
PARETO_FRONT_PATH = "prescriptors/pareto_front.png"
PARETO_FRONT = base64.b64encode(open(PARETO_FRONT_PATH, 'rb').read()).decode('ascii')

RANDOM_FOREST_PATH = "predictors/ELUC_forest.joblib"
