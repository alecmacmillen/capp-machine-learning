"""
CAPP 30254: Diagnostic assignment

Alec MacMillen
"""

import sys
import json
import csv
import pandas as pd
import util
import uszipcode

YEARS = [2017, 2018]


def query_crime(year, limit=500000):
    '''
    '''
    url = 'https://data.cityofchicago.org/resource/6zsd-86xi.json?'
    url += '$limit=' + str(limit) + "&$where=year=" + str(year)

    df = pd.read_json(url)
    valid_locations = df['location'].dropna()
    df = df.iloc[valid_locations.index]

    search = uszipcode.SearchEngine(simple_zipcode=True)
    df.loc[:, 'zipcode'] = df.apply(lambda x: find_zip(search, x['latitude'], x['longitude']), axis=1)
    #df.loc[:, 'zipcode'] = find_zip(search, df['latitude'], df['longitude']) 

    return df


def find_zip(search, latitude, longitude):
    '''
    '''
    result = search.by_coordinates(latitude, longitude)
    zip_code = result[0].zipcode
    return zip_code



def stack_crime_data(yearly_data):
    '''
    '''
    return pd.concat(yearly_data)


def write_out(df, output_path):
    '''
    '''
    with open(output_path, 'w') as of:
        df.to_csv(of, header=True, index=False)


def go(args):
    '''
    '''
    usage = ("")
    if len(sys.argv) != 2:
        print(usage)
        sys.exit(1)

    output_path = args[1]

    data_dict = {}
    for year in YEARS:
        data_dict[year] = query_crime(year)
    all_years = stack_crime_data(list(data_dict.values()))
    write_out(all_years, output_path)


if __name__=="__main__":
    go(sys.argv)
