"""
CAPP 30254: Assignment 1

Alec MacMillen
"""

import sys
import json
import csv
import pandas as pd

FEATURE_DICT = {"DP03_0062E":"med_hhinc",
                "DP02_0056PE":"pct_hs",
                "DP05_0038PE":"pct_blk",
                "DP04_0089E":"med_unitval"}

STATE = '17'
COUNTY = '031'
CRIME_PATH = 'crime_full.csv'

def pull_acs_data(state=STATE, county=COUNTY):
    '''
    '''
    url = "https://api.census.gov/data/2017/acs/acs5/profile?get="
    att = ','.join(FEATURE_DICT)
    url += att
    url += "&for=tract:*&in=state:" + state
    url += "&in=county:" + county

    acs_df = pd.read_json(url)
    headers = acs_df.iloc[0]
    acs_new = acs_df[1:]
    acs_new.columns = headers

    acs_new.rename(index=str, columns=FEATURE_DICT, inplace=True)
    acs_new['tract'] = acs_new['tract'].str[:4]

    return acs_new


def merge_on_tract(crime_data=CRIME_PATH):
    '''
    '''
    crime = pd.read_csv(crime_data)
    crime.loc[:, 'block_fix'] = crime['block'].str[:3] + '00' + crime['block'].str[5:]
    crime.loc[:, 'block_fix'] = crime['block_fix'].str.strip('0')

    crime['url'] = "https://geocoding.geo.census.gov/geocoder/geographies/address?street=" + \
        crime['block_fix'].str.split(' ').apply(lambda x: "+".join(x)) + "&city=Chicago&state=IL&benchmark=Public_AR_Census2010" + \
        "&vintage=Census2010_Census2010&layers=14&format=json"

    crime['tract_fix'] = crime['url'].apply(lambda x: extract_tract(x))

    return crime


def extract_tract(url):
    '''
    '''
    result = pd.read_json(url)
    location_dict = result.loc['addressMatches']['result']
    if location_dict:
        return location_dict[0]['geographies']['Census Blocks'][0]['TRACT']
    else:
        return "N/A"

