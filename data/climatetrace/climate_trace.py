import pandas as pd
import numpy as np

"""

Data is available from:

Downloads Page: https://climatetrace.org/downloads

"""

asset_generation_emissions = pd.read_csv("power/asset_electricity-generation_emissions.csv", header=0)
country_generation_emissions = pd.read_csv("power/country_electricity-generation_emissions.csv", header=0)

print(asset_generation_emissions.columns)
print(country_generation_emissions.columns)
positions = []
for row in asset_generation_emissions.iterrows():
    lon, lat = row[1]["st_astext"].split("POINT(")[-1].split(")")[0].split(" ")
    lon = float(lon)
    lat = float(lat)
    positions.append((lat, lon))

positions = np.asarray(positions)
