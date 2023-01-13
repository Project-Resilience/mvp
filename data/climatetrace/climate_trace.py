import pandas as pd
import numpy as np

"""

Data is available from:

Downloads Page: https://climatetrace.org/downloads

"""

asset_generation_emissions = pd.read_csv("power/asset_electricity-generation_emissions.csv", header=0)
country_generation_emissions = pd.read_csv("power/country_electricity-generation_emissions.csv", header=0)

