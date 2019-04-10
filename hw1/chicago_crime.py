"""
CAPP 30254: Diagnostic assignment

Alec MacMillen
"""

import sys
import pandas as pd
import uszipcode
import numpy as np

YEARS = [2017, 2018]


def query_crime(year, limit=500000):
    '''
    Query the Chicago open data API for all observations related to a given
    year. Dynamically construct the API query and read it as a JSON using
    Pandas, then keep only observations with valid location data (i.e.
    not null).

    Inputs: 
      year (int): the year for which data should be gathered
      limit (int): the max number of records to return with a single request

    Returns:
      df (pandas dataframe): pandas dataframe with all observations containing
        location data for the year specified
    '''
    url = 'https://data.cityofchicago.org/resource/6zsd-86xi.json?'
    url += '$limit=' + str(limit) + "&$where=year=" + str(year)

    df = pd.read_json(url)
    valid_locations = df['location'].dropna()
    df = df.iloc[valid_locations.index]

    return df


def find_zip(search, latitude, longitude):
    '''
    Use a "search" object created by the uszipcode module to determine the
    zip code associated with a given record using its latitude and longitude
    as lookup values.

    Inputs:
      search (uszipcode search object): initialized with uszipcode module's
        SearchEngine class
      latitude, longitude (floats): coordinate points

    Returns:
      zip_code: int (if exists, nan otherwise)
    '''
    result = search.by_coordinates(latitude, longitude, radius=5, returns=5)
    if result:
        zip_code = result[0].zipcode
    else:
        zip_code = np.nan

    return zip_code


def list_zips(df):
    '''
    Assign a zip code to every block. Aggregate dataframe with all block
    information to the block level, take the most common latitude and longitude
    associated with that block (the mode), and call find_zip function defined
    above to assign a zipcode to those coordinates.

    Inputs:
      df (pandas df): data frame with all crime information

    Returns:
      block_df (pandas df): data frame with zip codes merged on by block
    '''
    valid_location = pd.notnull(df['location'])
    df_valid = df[valid_location]
    blocks = df_valid.groupby(['block'])['latitude', 'longitude'] \
                     .agg(pd.Series.mode)
    block_df = blocks.reset_index()

    filt = block_df['latitude'].apply(lambda x: type(x) != np.float64)
    block_df.loc[filt, 'latitude'] = block_df['latitude'][0]
    block_df.loc[filt, 'longitude'] = block_df['longitude'][0]

    search = uszipcode.SearchEngine(simple_zipcode=True)
    block_df.loc[:, 'zipcode'] = block_df.apply(
        lambda x: find_zip(search, x['latitude'], x['longitude']), axis=1)

    return block_df


def stack_crime_data(yearly_data):
    '''
    Concatenate data frames from a list.

    Inputs:
      yearly_data (list): list of pandas dataframes

    Returns:
      concatenated data frame
    '''
    return pd.concat(yearly_data)


def write_out(df, output_path):
    '''
    Write dataframe out to csv file. 

    Inputs:
      df (pandas dataframe): dataframe to export
      output_path (str): file path of output file

    Returns:
      None
    '''
    with open(output_path, 'w') as of:
        df.to_csv(of, header=True, index=False)


def go(args):
    '''
    Execute program from the command line

    Inputs:
      args (command line arguments)

    Returns:
      None, outputs crime data with zipcodes merged on to
        csv file referenced in output_path
    '''
    usage = ("python chicago_crime.py <output_path>")
    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)

    output_path = args[1]

    data_dict = {}
    for year in YEARS:
        data_dict[year] = query_crime(year)
    all_years = stack_crime_data(list(data_dict.values()))

    zips = list_zips(all_years)
    block_zips = zips[['block', 'zipcode']]

    full = all_years.merge(block_zips, how='left', on='block')

    write_out(full, output_path)


if __name__ == "__main__":
    go(sys.argv)
