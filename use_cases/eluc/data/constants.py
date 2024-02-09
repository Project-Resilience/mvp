HF_PATH = "projectresilience/ELUC-committed"
# This should work if python path is set to use_cases/ELUC
DATA_FILE_PATH = "data/merged_aggregated_dataset_1850_2022.zarr.zip"
UPDATE_FILE_PATH = "data/BLUE_LUH2-GCB2022_ELUC-committed_gridded_net_1850-2021.nc"
# Country code conversion table from: https://gist.github.com/radcliff/f09c0f88344a7fcef373
# TODO: Note: This table is not perfect and has some errors, we should consider manually fixing them.
# I tried my best but I'm not 100% sure it's correct.
CODES_PATH = "data/codes.csv"

LAND_USE_COLS = ['c3ann', 'c3nfx', 'c3per','c4ann', 'c4per', 
                 'pastr', 'primf', 'primn', 
                 'range', 'secdf', 'secdn', 'urban']
CROP_COLS = ['c3ann', 'c3nfx', 'c3per','c4ann', 'c4per']
LAND_USE_COLS = ["crop"] + [col for col in LAND_USE_COLS if col not in CROP_COLS]
DIFF_LAND_USE_COLS = [f"{col}_diff" for col in LAND_USE_COLS]
COLS_MAP = dict(zip(LAND_USE_COLS, DIFF_LAND_USE_COLS))

NO_CHANGE_COLS = ["c4per_diff", "primf_diff", "primn_diff", "urban_diff"]
RECO_COLS = [col.split("_")[0] for col in DIFF_LAND_USE_COLS if col not in NO_CHANGE_COLS]
DIFF_RECO_COLS = [f"{col}_diff" for col in RECO_COLS]
RECO_MAP = dict(zip(RECO_COLS, DIFF_RECO_COLS))

NONLAND_FEATURES = ["cell_area", "lat", "lon", "time"]

NN_FEATS = LAND_USE_COLS + NONLAND_FEATURES + DIFF_LAND_USE_COLS

# ["United Kingdom", "France", "Germany", "Netherlands", "Belgium", "Switzerland", "Ireland"]
EU_COUNTRIES = ["GB", "FR", "DE", "NL", "BE", "CH", "IE"]
# ["Brazil", "Bolivia", "Paraguay", "Peru", "Ecuador", "Colombia", "Venezuela", "Guyana", "Suriname", "Uruguay", "Argentina", "Chile"]
SA_COUNTRIES = ["BR", "BO", "PY", "PE", "EC", "CO", "VE", "GY", "SR", "UY", "AR", "CL"]
# ["United States"]
US_COUNTRIES = ["US"]
COUNTRY_DICT = {"EU": EU_COUNTRIES, "SA": SA_COUNTRIES, "US": US_COUNTRIES, "ALL": None}

CAO_MAPPING = {'context': LAND_USE_COLS + NONLAND_FEATURES, 
               'actions': DIFF_LAND_USE_COLS, 
               'outcomes': ["ELUC", "change"]}
