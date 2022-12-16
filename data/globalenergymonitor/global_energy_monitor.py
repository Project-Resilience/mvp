import pandas as pd
import numpy as np

"""

Data is available from:

Coal: https://globalenergymonitor.org/projects/global-coal-plant-tracker/
Gas: https://globalenergymonitor.org/projects/global-gas-plant-tracker/
Solar: https://globalenergymonitor.org/projects/global-solar-power-tracker/
Wind: https://globalenergymonitor.org/projects/global-wind-power-tracker/

Changes in Coal Tracker: https://drive.google.com/drive/folders/1kbLck8dEWlqUifv98AHNgL3KA_wMf1nL?usp=sharing

"""

solar_data = pd.read_excel("Global-Solar-Power-Tracker-May-2022.xlsx", sheet_name=None)
solar_data = solar_data['Data']
wind_data = pd.read_excel("Global-Wind-Power-Tracker-May-2022.xlsx", sheet_name=None)
wind_data = wind_data['Data']
gas_data = pd.read_excel("Global-Gas-Plant-Tracker-Aug-2022.xlsx", sheet_name=None)
coal_data = pd.read_excel("Global-Coal-Plant-Tracker-July-2022.xlsx", sheet_name=None)
coal_changes = pd.read_excel("July 2022 GCPT Status Changes - 2014 - 2022.xlsx", sheet_name=None)
print(coal_changes)
exit()
"""
Things to extract and put in database:
Context: 
Fraction of energy from Coal per region
Fraction of energy from Gas per region

Action:
Proposed Transition To (%):
- Solar
- Wind
- Gas

For Q1-4 data summary -> Also have it do more fine grained (per month?) as well

For quarterly data -> line of region (large region) with number of plants of each type and total generating capacity
Same for future, proposed number of plants and generating capacity for the next quarters as appropriate
For region, and (subnational unit, State/Province, as they are the same it seems), and Country
Include coal-to-gas conversion/replacement in there as well
 
"""

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

def extract_and_format_wind_solar(solar, wind):
    drop_columns = ["Project Name in Local Language / Script", "Operator", "Operator Name in Local Language / Script",
                    "Owner Name in Local Language / Script", "Owner", "GEM phase ID", "GEM location ID", "Wiki URL",
                    "Other IDs (location)", "Other IDs (unit/phase)", "Other Name(s)"]
    solar = solar.drop(columns=drop_columns)
    wind = wind.drop(columns=drop_columns)
    grouped_solar = solar.groupby(['State/Province','Start year', 'Status']).agg(
        sum_solar_capacity_mw=('Capacity (MW)', 'sum'),
    ).reset_index()
    grouped_wind = wind.groupby(['State/Province','Start year', 'Status']).agg(
        sum_wind_capacity_mw=('Capacity (MW)', 'sum'),
    ).reset_index()
    return solar, grouped_solar, wind, grouped_wind

solar, grouped_solar, wind, grouped_wind = extract_and_format_wind_solar(solar_data, wind_data)
gas, grouped_gas = extract_and_format_gas(gas_data)
coal, grouped_coal = extract_and_format_coal(coal_data)

# Combine on region name the MW generation by type
from functools import reduce
combined_grouped = reduce(lambda x,y: pd.merge(x,y, on=['State/Province', 'Status', 'Start year'], how='outer'), [grouped_solar, grouped_wind, grouped_coal, grouped_gas])
combined_grouped["sum_capacity_mw"] = combined_grouped['sum_gas_capacity_mw'].fillna(0.) + combined_grouped['sum_coal_capacity_mw'].fillna(0.) + combined_grouped['sum_solar_capacity_mw'].fillna(0.) + combined_grouped['sum_wind_capacity_mw'].fillna(0.)

# Now need to know per year what the percentage is, not the start year, but for all ones after start year, and operational, what is percentage then

combined_grouped["percentage_mw_gas"] = combined_grouped['sum_gas_capacity_mw'].fillna(0.) / combined_grouped["sum_capacity_mw"]
combined_grouped["percentage_mw_coal"] = combined_grouped['sum_coal_capacity_mw'].fillna(0.) / combined_grouped["sum_capacity_mw"]
combined_grouped["percentage_mw_wind"] = combined_grouped['sum_wind_capacity_mw'].fillna(0.) / combined_grouped["sum_capacity_mw"]
combined_grouped["percentage_mw_solar"] = combined_grouped['sum_solar_capacity_mw'].fillna(0.) / combined_grouped["sum_capacity_mw"]

print(combined_grouped)
print(combined_grouped.columns)

grouped_production = combined_grouped.groupby(['State/Province', 'Status']).agg(
        region_mw_gas_total=('sum_gas_capacity_mw', np.nansum),
        region_mw_coal_total=('sum_coal_capacity_mw', np.nansum),
        region_mw_wind_total=('sum_wind_capacity_mw', np.nansum),
        region_mw_solar_total=('sum_solar_capacity_mw', np.nansum),
    ).reset_index()

grouped_production["region_total_mw"] = grouped_production['region_mw_gas_total'].fillna(0.) + grouped_production['region_mw_coal_total'].fillna(0.) + grouped_production['region_mw_wind_total'].fillna(0.) + grouped_production['region_mw_solar_total'].fillna(0.)

grouped_production["percentage_mw_gas"] = grouped_production['region_mw_gas_total'].fillna(0.) / grouped_production["region_total_mw"]
grouped_production["percentage_mw_coal"] = grouped_production['region_mw_coal_total'].fillna(0.) / grouped_production["region_total_mw"]
grouped_production["percentage_mw_wind"] = grouped_production['region_mw_wind_total'].fillna(0.) / grouped_production["region_total_mw"]
grouped_production["percentage_mw_solar"] = grouped_production['region_mw_solar_total'].fillna(0.) / grouped_production["region_total_mw"]

selected_production = grouped_production[(grouped_production['percentage_mw_wind'] > 0) & (grouped_production['percentage_mw_solar'] > 0) & (grouped_production['percentage_mw_gas'] > 0)].dropna()
