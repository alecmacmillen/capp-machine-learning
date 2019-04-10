## README

To replicate results:

1. Make sure all packages listed in requirements.txt are installed.

2. Run chicago_crime.py to output crime data pulled down from Chicago open data portal.
Example run: python chicago_crime.py crime.csv
This will output a CSV called "crime.csv" to the working directory.

3. Run acs.py to pull down ACS data by zipcode, merge with crime data on zip and output full data.
Example run: python acs.py acs_crime_merged.csv acs_full.csv crime.csv
This will read in crime.csv, pull down ACS data from the Census Bureau API, merge on zip code, and output a full merged dataset.
If this file outputs an HTTP Error 500: Internal Server Error, wait a few minutes and try re-running (appears to have something to do with the ACS website).

4. Run "Chicago Crime 2017-2018 Analysis (CAPP 30254 HW1).ipynb" for visualizations and analysis. Writeups also found in this file.
The Jupyter notebook reads in CSV files output by the .py scripts as Pandas dataframes.