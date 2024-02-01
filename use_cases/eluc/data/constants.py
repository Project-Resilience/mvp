import pandas as pd
import regionmask

# TODO: This has to be changed for your local machine
DATA_FILE_PATH = "../../../mvp/data/gcb/merged_aggregated_dataset_1850_2022.zarr.zip"
UPDATE_FILE_PATH = "../../../mvp/data/gcb/BLUE_LUH2-GCB2022_ELUC-committed_gridded_net_1850-2021.nc"

LAND_USE_COLS = ['c3ann', 'c3nfx', 'c3per','c4ann', 'c4per', 'pastr', 'primf', 'primn', 'range', 'secdf', 'secdn', 'urban']

# TODO: Temporary to merge the crops together
CROP_COLS = ['c3ann', 'c3nfx', 'c3per','c4ann', 'c4per']
LAND_USE_COLS = ["crop"] + [col for col in LAND_USE_COLS if col not in CROP_COLS]

DIFF_LAND_USE_COLS = [f"{col}_diff" for col in LAND_USE_COLS]
COLS_MAP = dict(zip(LAND_USE_COLS, DIFF_LAND_USE_COLS))

# RECO_COLS = ['c3ann', 'c3nfx', 'c3per','c4ann', 'pastr', 'range', 'secdf', 'secdn']
NO_CHANGE_COLS = ["c4per_diff", "primf_diff", "primn_diff", "urban_diff"]
RECO_COLS = [col.split("_")[0] for col in DIFF_LAND_USE_COLS if col not in NO_CHANGE_COLS]
DIFF_RECO_COLS = [f"{col}_diff" for col in RECO_COLS]
RECO_MAP = dict(zip(RECO_COLS, DIFF_RECO_COLS))

NONLAND_FEATURES = ["cell_area", "lat", "lon", "time"]

# TODO: Updated this to include lat lon time
# NN_FEATS = LAND_USE_COLS + ["cell_area"] + DIFF_LAND_USE_COLS
NN_FEATS = LAND_USE_COLS + NONLAND_FEATURES + DIFF_LAND_USE_COLS

EU_COUNTRIES = ["GB", "FR", "DE", "NL", "BE", "CH", "IE"]
# ["Brazil", "Bolivia", "Paraguay", "Peru", "Ecuador", "Colombia", "Venezuela", "Guyana", "Suriname", "Uruguay", "Argentina", "Chile"]
SA_COUNTRIES = ["BR", "BO", "PY", "PE", "EC", "CO", "VE", "GY", "SR", "UY", "AR", "CL"]
US_COUNTRIES = ["US"]
COUNTRY_DICT = {"EU": EU_COUNTRIES, "SA": SA_COUNTRIES, "US": US_COUNTRIES, "ALL": None}

MANUAL_MAP = {
    "INDO": 360,
    "DRC": 180,
    "RUS": 643,
    "N": 578,
    "F": 250,
    "J": 388,
    "NA": 516,
    "PAL": 275,
    "J": 400,
    "IRQ": 368,
    "IND": 356,
    "IRN": 364,
    "SYR": 760,
    "ARM": 51,
    "S": 752,
    "A": 36,
    "EST": 233,
    "D": 276,
    "L": 442,
    "B": 56,
    "P": 620,
    "E": 724,
    "IRL": 372,
    "I": 380,
    "SLO": 705,
    "FIN": 246,
    "J": 392,
    "BiH": 70,
    "NM": 807,
    "KO": 383,
    "SS": 728
}

countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
COUNTRIES_DF = countries.to_dataframe()

codes_df = pd.read_csv("../../../mvp/data/gcb/conversion/codes.csv")

# Replace all the bad codes with their real ones
for i in range(len(COUNTRIES_DF)):
    old_abbrev = COUNTRIES_DF.iloc[i]["abbrevs"]
    if old_abbrev in MANUAL_MAP.keys() and MANUAL_MAP[old_abbrev] in codes_df["Numeric code"].unique():
        COUNTRIES_DF.iloc[i]["abbrevs"] = codes_df[codes_df["Numeric code"] == MANUAL_MAP[old_abbrev]]["Alpha-2 code"].iloc[0]

CAO_MAPPING = {'context': LAND_USE_COLS + NONLAND_FEATURES, 'actions': DIFF_LAND_USE_COLS, 'outcomes': ["ELUC", "change"]}
#CAO_MAPPING = {'context': ['c3ann', 'c3nfx', 'c3per', 'c4ann', 'c4per', 'pastr', 'primf', 'primn', 'range', 'secdf', 'secdn', 'urban', 'cell_area'], 'actions': ['c3ann_diff', 'c3nfx_diff', 'c3per_diff', 'c4ann_diff', 'c4per_diff', 'pastr_diff', 'primf_diff', 'primn_diff', 'range_diff', 'secdf_diff', 'secdn_diff', 'urban_diff'], 'outcomes': ['ELUC', 'change']}