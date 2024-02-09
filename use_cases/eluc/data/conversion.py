import pandas as pd
import regionmask

from data import constants

MANUAL_MAP = {
    "INDO": 360,
    "DRC": 180,
    "RUS": 643,
    "N": 578,
    "F": 250,
    "J": 388,
    "NA": 516,
    "PAL": 275,
    # "J": 400,
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

def construct_countries_df():
    """
    Constructs a dataframe mapping of countries, their abbreviations, and their proper codes.
    """
    countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
    countries_df = countries.to_dataframe()

    codes_df = pd.read_csv(constants.CODES_PATH)

    # Replace all the bad codes with their real ones
    for i in range(len(countries_df)):
        old_abbrev = countries_df.iloc[i]["abbrevs"]
        if old_abbrev in MANUAL_MAP.keys() and MANUAL_MAP[old_abbrev] in codes_df["Numeric code"].unique():
            countries_df.iloc[i]["abbrevs"] = codes_df[codes_df["Numeric code"] == MANUAL_MAP[old_abbrev]]["Alpha-2 code"].iloc[0]

    return countries_df