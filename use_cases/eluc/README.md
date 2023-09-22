# Emissions from Land Use Change (ELUC)

Goal: provide decision makers with tools to know how their land-use choices affect CO2 fluxes in the long-term.  

The tools should help decisions makers with their choices:
- For a geographical grid cell, identified by its latitude and longitude, what changes to the land usage can be
made to reduce CO2 emissions?
- What will be the long term CO2 impact of changing the land usage in a particular way?

It is possible to learn from historical decisions made by decision makers all around the world if they can be compared. 

## Data

### ELUC

BLUE simulations with committed emissions could be used to estimate the long-term CO2 impact.
"Committed emissions" means all the emissions that are caused by a land-use change event are attributed to the year
of the event.
BLUE (bookkeeping of land use emissions) is a bookkeeping model that attributes carbon fluxes to land use activities.
See [(BLUE: Bookkeeping of land use emissions)](https://doi.org/10.1002/2014GB004997) for more details.  

The team in charge of the BLUE model performed such simulations with BLUE and generated the file
`BLUE_LUH2-GCB2022_ELUC-committed_gridded_net_1850-2021.nc` available in
[this shared folder](https://syncandshare.lrz.de/getlink/fiAuJz5VFgsEJ1E96mthCT/Data_GCB2022).

### LUC

The Land Use Change (LUC) data is provided by the LUH2 project [(LUH2: Land Use Harmonization 2)](https://luh.umd.edu/)
The land-use harmonization strategy estimates the fractional land-use patterns, underlying land-use transitions,
and key agricultural management information, annually for the time period 850-2100 at 0.25 x 0.25 resolution.  

The LUH2 model and datasets prepared for CMIP6 are described in
[Hurtt et al. 2020.](https://gmd.copernicus.org/articles/13/5425/2020/gmd-13-5425-2020-discussion.html)

Land-use Harmonization (LUH) data for [GCB 2022](https://doi.org/10.5194/essd-14-4811-2022) is provided in
3 separate files, which can be downloaded from the following links
(for the states, transitions, and management data layers respectively):

http://luh.umd.edu/LUH2/LUH2_GCB_2022/states.nc  
http://luh.umd.edu/LUH2/LUH2_GCB_2022/transitions.nc  
http://luh.umd.edu/LUH2/LUH2_GCB_2022/management.nc  

These files are based on the new HYDE3.3, as well as the 2021 FAO wood harvest data, for all years 850-2022.

The data files are for the years 850-2022, which keeps the file format consistent with the LUH2 data produced for CMIP6,
hence the start year of 850. The LUH2-GCB2022 data will be different from the LUH2 v2h data used for CMIP6 for all
years, due to the use of the new HYDE3.3 crop/grazing land dataset.

See https://luh.umd.edu/  for more details.

### Land Use Types

Primary: Vegetation that is untouched by humans
- **primf**: Primary forest
- **primn**: Primary nonforest vegetation

- Secondary: Vegetation that has been touched by humans
- **secdf**: Secondary forest
- **secdn**: Secondary nonforest vegetation

Urban
- **Urban**: Urban areas

Crop
- **c3ann**: Annual C3 crops (e.g. wheat)
- **c4ann**: Annual C4 crops (e.g. maize)
- **c3per**: Perennial C3 crops (e.g. banana)
- **c4per**: Perennial C4 crops (e.g. sugarcane)
- **c3nfx**: Nitrogen fixing C3 crops (e.g. soybean)

Pasture
- **pastr**: Managed pasture land
- **range**: Natural grassland / savannah / desert / etc.


## Modeling decisions

A decision can be represented by 3 constituents: **context**, **actions** and **outcomes**

> In a particular **context**, what was the **outcomes** of the decision maker **actions**?

In other words, in a particular situation, what were the results of the decisions of the decision maker?

### Context

The context describes the current situation the decision maker is in.
For the land use change use case this is a particular grid cell,
a point in time when the decision had to be made, and the current usage of the land:

- Latitude
- Longitude
- Area
- Year
- Land usage, as a percentage, summing up to 100%
  - primf
  - primn
  - sedf
  - secdn
  - urban
  - c3ann
  - c4ann
  - c3per
  - c4per
  - c3fnx
  - pastr
  - range
  - nonland

Latitude and longitude represent the cell on the grid.  
Area represents the surface of the cell. Cells close to the equator have a bigger area than cells close to the poles.  
Year is useful to capture historical decisions: the same cell has been through a lot of land use changes
over the years.  
Land usage represents the percentage of the land used by each land type. Note there is a 'nonland' type that represents
the percentage of the cell that is not land (e.g typically sea, lake, etc.). The land usage types sum up to 100%. 

### Actions

Actions represent the decisions decision makers can make. How can they change the land?
We considered 2 limitations:
1. Decision makers can't affect primary land:
   1. It's better to preserve primary vegetation. Destroying it is not an option.
   2. It's not possible to re-plant primary vegetation. Once destroyed, it's destroyed forever and
   can't be planted back (it would become secondary vegeation)
2. Decision makers can't affect urban areas. We consider the needs for larger / smaller urban areas are dictated by
   other imperatives, other decision makers.
   1. It doesn't seem reasonable to recommend to destroy a city
   2. It doesn't seem reasonable to recommend to expand a city 

### Outcomes

- Emissions from Land Use Change (ELUC): CO2 emissions, in metric ton of carbon per hectare (tC/ha),
resulting from the land use change
- Percentage of land that was changed

There is a trade-off between these 2 objectives: it easy to reduce emissions by changing
most of the land, but that would come at a huge cost. This "cost" can be approximately derived from
the percentage of land that was change. In other words decision makers have to:
- minimize ELUC
- while minimizing land change at the same time

## Modeling

### Predictions

Given **context** and **actions** -> predict **outcomes**

This is a prediction problem.

Anyone can contribute a prediction model, as long as it complies with the `predict` interface
and its inputs and outputs.

TODO: point to code that trains predictor models

### Prescriptions

Given **context** -> prescribe **actions** that optimize **outcomes**

This is an optimization problem.
Anyone can contribute a prescription model, as long as it complies with the `prescribe` interface
and its inputs and outputs.

## Robojudge

"Robojudge" is an interactive tool that can be used to compare predictors.
Some models perform better depending on the evaluation metric, the countries or the years on which they are evaluated.   

## Ensembling

Ensemble models can be constructed from predicition models.

## Demo

A user interface for decision makers is available here: https://landuse.evolution.ml/

See [demo/README.md](demo/README.md])

## References

ELUC data provided by the BLUE model [(BLUE: Bookkeeping of land use emissions)](https://doi.org/10.1002/2014GB004997)
Land use change data provided by the LUH2 project [(LUH2: Land Use Harmonization 2)](https://luh.umd.edu/)
Setup is described in Appendix C2.1 of the GCB 2022 report [(Global Carbon Budget 2022 report)](https://doi.org/10.5194/essd-14-4811-2022)
The Global Carbon Budget report assesses the global CO2 budget for the Intergovernmental Panel on Climate Change [(IPCC)](https://www.ipcc.ch/)
