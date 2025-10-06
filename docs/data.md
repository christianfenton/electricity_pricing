# Data Pipeline

## Instructions

Processed weather and electricity price data is already available in this repository's `data/processed/` directory. Users that want to  process raw data themselves can download the relevant raw weather and gas price data from the Met Office and Office for National Statistics websites, update the paths in the `create_datasets.py` accordingly, and run
```bash
python create_datasets.py
```

## Helpful Information

### Weather data locations

- Heathrow, Greater London: Close to major population centre and wind farms in the Thames Estuaray
- Crosby, Merseyside: Close to major population centre and wind farms in the Irish Sea
- Dyce, Aberdeenshire: Close to major population centre and wind farms in the North Sea

The raw weather data can be found on the Met Office MIDAS archive at:
- Heathrow: https://data.ceda.ac.uk/badc/ukmo-midas-open/data/uk-hourly-weather-obs/dataset-version-202507/greater-london/00708_heathrow
- Crosby: https://data.ceda.ac.uk/badc/ukmo-midas-open/data/uk-hourly-weather-obs/dataset-version-202507/merseyside/17309_crosby
- Dyce: https://data.ceda.ac.uk/badc/ukmo-midas-open/data/uk-hourly-weather-obs/dataset-version-202507/aberdeenshire/00161_dyce

### Units

- Power: megawatts (MW)
- Electricity and natural gas prices: GBP per megawatt hour (£ / MWh)
- Wind speed: metres per second (m/s)
- Wind direction: degrees
- Temperature: degrees Celsius
- Solar irradiation: kilojoules per metre squared (KJ/m^2)

### Glossary of terms

- AGPT: Actual generation data per settlement period aggregrated by power system resource type
- FUELHH: Half-hourly generation outturn aggregrated by fuel type
- CCGT: Combined cycle gas turbine
- OCGT: Open cycle gas turbine
- NPSHYD: Non-pumped storage hydropower
- PS: Pumped storage
- INTER:  Imports/exports from/to other grids via interconnectors
- INDO: Initial national demand outturn.
- ITSDO: Initial transmission system demand outturn.

### Data processing

The AGPT data does not include flows from interconnects, while the FUELHH data does not include energy generation from solar or embedded generation. The AGPT and FUELHH data are merged to get an accurate breakdown of the different energy generation sources.

## References

- Elexon BMRS API: https://bmrs.elexon.co.uk/api-documentation/introduction
- Met Office (2025): MIDAS Open: UK hourly weather observation data, v202507. NERC EDS Centre for Environmental Data Analysis, 18 July 2025. doi:10.5285/99173f6a802147aeba430d96d2bb3099.
- Met Office (2025): MIDAS Open: UK hourly solar radiation data, v202507. NERC EDS Centre for Environmental Data Analysis, 18 July 2025. doi:10.5285/76e54f87291c4cd98c793e37524dc98e.
- Natural gas prices: https://www.ons.gov.uk/economy/economicoutputandproductivity/output/datasets/systemaveragepricesapofgas