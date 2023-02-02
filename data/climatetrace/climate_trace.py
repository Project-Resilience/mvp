import pandas as pd
import numpy as np
from scipy import spatial

def extract_and_format_gas(gas):
    drop_columns = ['Wiki URL', 'Wiki URL local language',
       'Plant name (local script)', 'Owner', 'Parent', 'Other IDs (location)',
       'Other IDs (unit)', 'Other plant names', 'Captive [heat, power, both]',
       'Captive industry type',
       'Captive non-industry use [heat, power, both, none]',
       'GEM location ID',
       'GEM unit ID']
    gas = gas['Gas plants - data'].drop(columns=drop_columns).fillna({"Coal-to-gas conversion/replacement?": "Unknown", "CCS attachment?": "Unknown", "Hydrogen capable?": "Unknown"}).rename(columns={"Subnational unit (province, state)": "State/Province"})
    grouped_gas = gas.groupby(["State/Province", 'Start year', 'Fuel', 'Status', 'Coal-to-gas conversion/replacement?', 'CCS attachment?', 'Hydrogen capable?']).agg(
        sum_gas_capacity_mw=('Capacity elec. (MW)', 'sum'),
    ).reset_index()
    return gas, grouped_gas

def extract_and_format_coal(coal):
    drop_columns = ['Tracker ID', 'TrackerLOC', 'ParentID', 'Wiki page', 'Chinese Name',
       'Other names', 'Owner', 'Parent', 'Permits', 'Captive', 'Captive industry use',
       'Captive residential use', 'Heat rate (Btu per kWh)', 'Capacity factor',]

    coal = coal['Units'].drop(columns=drop_columns).fillna({"Coal type": "Unknown"}).rename(columns={"Subnational unit (province, state)": "State/Province", "Year": "Start year"})
    grouped_coal = coal.groupby(
        ['State/Province','Start year', 'Coal type', 'Status']).agg(
        sum_coal_capacity_mw=('Capacity (MW)', 'sum'),
    ).reset_index()
    return coal, grouped_coal

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

# Now need to find matching ones from the Global Energy Monitor dataset
gas_data = pd.read_excel("/home/jacob/Development/mvp/data/globalenergymonitor/Global-Gas-Plant-Tracker-Aug-2022.xlsx", sheet_name=None)
coal_data = pd.read_excel("/home/jacob/Development/mvp/data/globalenergymonitor/Global-Coal-Plant-Tracker-July-2022.xlsx", sheet_name=None)
gas, _ = extract_and_format_gas(gas_data)
coal, _ = extract_and_format_coal(coal_data)

print(gas.columns)
print(coal.columns)

gas_positions = []
for row in gas.iterrows():
    data = row[1]
    gas_positions.append((float(data["Latitude"]), float(data["Longitude"])))

gas_positions = np.asarray(gas_positions)

coal_positions = []
for row in coal.iterrows():
    data = row[1]
    coal_positions.append((float(data["Latitude"]), float(data["Longitude"])))

coal_positions = np.asarray(coal_positions)

print(f"Sizes: ClimateTrace: {len(positions)} Coal: {len(coal_positions)} Gas: {len(gas_positions)}")

# Build KDTrees and search
climate_trace_tree = spatial.KDTree(positions)

# Now go through gas and coal, and find ones that combine
coal_matches = []
for i, pt in enumerate(coal_positions):
    distance, index = climate_trace_tree.query(pt)
    if distance < 0.01:
        coal_matches.append((i, index))

coal_matches = np.asarray(coal_matches)
#print(coal_matches)
print(f"There is a {len(coal_matches)/len(coal_positions) * 100}% ({len(coal_matches)}/{len(coal_positions)}) overlap between ClimateTrace and Global Energy Monitor Coal Tracker")


gas_matches = []
for i, pt in enumerate(gas_positions):
    distance, index = climate_trace_tree.query(pt)
    if distance < 0.01:
        gas_matches.append((i, index))

gas_matches = np.asarray(gas_matches)
#print(gas_matches)
print(f"There is a {len(gas_matches)/len(gas_positions) * 100}% ({len(gas_matches)}/{len(gas_positions)}) overlap between ClimateTrace and Global Energy Monitor Gas Tracker")
