"""
CAPP 30254: Assignment 1

Alec MacMillen
"""

import sys
import pandas as pd

FEATURE_DICT = {"DP03_0062E":"med_hhinc",
                "DP02_0056PE":"pct_hs",
                "DP05_0038PE":"pct_blk",
                "DP04_0089E":"med_unitval"}

CRIME_DATA = "all_crime_stats.csv"


def pull_acs_data(features=FEATURE_DICT):
    '''
    Pull down ACS data by reading a JSON request from the ACS API with defined
    features, convert first row of data to header, rename columns with labels
    that are descriptive.

    Inputs:
      features (dict): dictionary mapping column names as they appear in 
        ACS data to more descriptive names

    Returns:
      acs_new (pandas df): summary ACS statistics for selected fields at the
        zip code level. Includes all zip codes in the US.
    '''
    url = "https://api.census.gov/data/2017/acs/acs5/profile?get="
    att = ','.join(features)
    url += att
    url += "&for=zip%20code%20tabulation%20area:*"

    acs_df = pd.read_json(url)
    headers = acs_df.iloc[0]
    acs_new = acs_df[1:]
    acs_new.columns = headers

    acs_new.rename(index=str, columns=features, inplace=True)
    acs_new.rename(
        columns={'zip code tabulation area':'zipcode'}, inplace=True)

    return acs_new


def go(args):
    '''
    Pull down ACS data and merge it on to existing Chicago crime data on
    zipcode. Write the data out to a CSV for use in analysis.

    Inputs:
      args (from command line)

    Returns:
      writes out full merged dataset to output path
    '''
    usage = ("python acs.py <merged_output_path>"
             "<acs_output_path> <crime_input_data>")
    if len(sys.argv) != 4:
        print(usage)
        sys.exit(1)

    merged_output_path = args[1]
    acs_output_path = args[2]
    crime_data = args[3]

    acs_data = pull_acs_data()
    acs_data.to_csv(acs_output_path, header=True, index=False)

    crime_df = pd.read_csv(crime_data)
    crime_df['zipcode'] = crime_df['zipcode'].astype(str)
    final_data = crime_df.merge(acs_data, how='left', on='zipcode')

    final_data.to_csv(merged_output_path, header=True, index=False)


if __name__ == "__main__":
    go(sys.argv)
    