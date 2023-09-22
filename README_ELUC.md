# Emissions from Land Use Change (ELUC)

Goal: provide decision makers with tools to know how their land-use choices affect CO2 fluxes in the long-term.  

The tools should help decisions makers with their choices:
- For a geographical grid cell, identified by its latitude and longitude, what changes to the land usage can be
made to reduce CO2 emissions?
- What will be the long term CO2 impact of changing the land usage in a particular way?

## Data

### ELUC

BLUE simulations with committed emissions could be used to estimate the long-term CO2 impact.
Committed emissions: all the emissions that are caused by a land-use change event are attributed to the year of the event
BLUE (bookkeeping of land use emissions) is a bookkeeping model that attributes carbon fluxes to land use activities.
See [(BLUE: Bookkeeping of land use emissions)](https://doi.org/10.1002/2014GB004997) for more details.  

The team in charge of the BLUE model performed such simulations with BLUE and generated the file
`BLUE_LUH2-GCB2022_ELUC-committed_gridded_net_1850-2021.nc` available in
[this shared folder](https://syncandshare.lrz.de/getlink/fiAuJz5VFgsEJ1E96mthCT/Data_GCB2022).

### LUC

The Land Use Change (LUC) data provided by the LUH2 project [(LUH2: Land Use Harmonization 2)](https://luh.umd.edu/)

## Modeling

### Context

The context is the current usage of the land

### Actions

### Outcomes

1. Emissions from Land Use Change (ELUC): CO2 emissions, in metric ton of carbon per hectare (tC/ha),
resulting from the land use change
2. Percentage of land that was changed

## References

ELUC data provided by the BLUE model [(BLUE: Bookkeeping of land use emissions)](https://doi.org/10.1002/2014GB004997)
Land use change data provided by the LUH2 project [(LUH2: Land Use Harmonization 2)](https://luh.umd.edu/)
Setup is described in Appendix C2.1 of the GCB 2022 report [(Global Carbon Budget 2022 report)](https://essd.copernicus.org/articles/14/4811/2022/#section10/)
The Global Carbon Budget report assesses the global CO2 budget for the Intergovernmental Panel on Climate Change [(IPCC)](https://www.ipcc.ch/)
