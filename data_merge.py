'''
Data that we need:
JHU county level data
JHU state level testing rates
BLS county level unemployment
ACS:
    - total population
    - race
    - population density
    - internet access
    - educational attainment
    - median family income
    - prevalence of occupation types
NYT stay at home order dates
KFF governor's party
Political lean/2016 election split

'''

'''
ACS VARIABLE NAMES
B02001_001E - total pop
B02001_002E - white pop
B02001_003E - black pop
B02001_004E - American Indian/Alaska Native pop
B02001_005E - Asian pop
B03001_002E - non hispanic pop
B03001_003E - hispanic pop

B28002_013E - no internet

B15003_022E - bachelor's degree
B23006_023E - bachelor's degree or higher

B19013_001E - median family income

C24050_002E - agriculture, forestry, fishing and hunting, mining
C24050_003E - construction
C24050_004E - manufacturing
C24050_005E - wholesale trade
C24050_006E - retail trade
C24050_007E - transportation and warehousing
C24050_008E - information
C24050_009E - finance, insurance, real estate
C24050_010E - professional, scientific, and management
C24050_011E - education services, health care, social assistance
C24050_012E - arts, entertainment, recreation, accomodation, food services
C24050_013E - other
C24050_014E - public administration
C24060_002E - management, business, science, and arts
C24060_003E - service occupations
C24060_004E - sales and office occupations
C24060_005E - natural resources, construction, maintenance
C24060_006E - production, transportation, and material moving



'''

import requests
import pandas as pd
import numpy as np

#Constants
#The internet saved me on this one: https://gist.github.com/rogerallen/1583593
US_STATE_ABBREV = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}


#Import and clean ACS and Census data
HOST = "https://api.census.gov/data"
year = "2018"
dataset = "acs/acs5"
base_url = "/".join([HOST, year, dataset])
predicates = {}
get_vars = ['NAME', 'B02001_001E', 'B02001_002E', 'B02001_003E', 'B03001_003E',
            'B28002_013E', 'B15003_022E', 'B23006_023E', 'B19013_001E',
            'C24050_012E']
predicates["get"] = ",".join(get_vars)
predicates["for"] = "county:*"
r = requests.get(base_url, params=predicates)

df = pd.DataFrame(data=r.json())
df.columns = df.iloc[0]
df = df[1:]
df['county_name'] = df['NAME'].str.extract('(.*) County')
df['state_name'] = df['NAME'].str.extract(', (.*)')

#Clean columns and column names
#Change column types to int
cols_to_change = ['B02001_001E', 'B02001_002E', 'B02001_003E', 'B03001_003E',
            'B28002_013E', 'B15003_022E', 'B23006_023E', 'B19013_001E',
            'C24050_012E']
for col in cols_to_change:
    df[col] = pd.to_numeric(df[col])

#Race
df.rename({'B02001_001E': 'total_pop', 'B19013_001E': 'median_income'}, axis=1, inplace=True)
df['prop_white'] = df['B02001_002E'] / df['total_pop']
df['prop_black'] = df['B02001_003E'] / df['total_pop']
df['prop_hisp'] = df['B03001_003E'] / df['total_pop']

df['log_med_income'] = np.log(df['median_income'])

#Internet access
df['prop_no_internet'] = df['B28002_013E'] / df['total_pop']

#Education
df['prop_ba'] = df['B15003_022E'] / df['total_pop']

#Occupation
df['prop_services'] = df['C24050_012E'] / df['total_pop']

#Drop extra columns
df_clean = df.drop(['B02001_002E', 'B02001_003E', 'B03001_003E',
            'B28002_013E', 'B15003_022E', 'B23006_023E', 'C24050_012E'], axis=1)


#Get population density
base_url = 'https://api.census.gov/data/2018/pep/population'
predicates = {}
get_vars = ['DENSITY']
predicates["get"] = ",".join(get_vars)
predicates["for"] = "county:*"
r2 = requests.get(base_url, params=predicates)

density = pd.DataFrame(data=r2.json())
density.columns = density.iloc[0]
density = density[1:]

df_merged = df_clean.merge(density, on=['state', 'county'])
df_merged.rename({'NAME': 'name', 'DENSITY': 'pop_density'}, axis=1, inplace=True)
df_merged['FIPS'] = (df_merged['state'].astype(
    str) + df_merged['county'].astype(str).str.pad(3, fillchar='0')).astype(float)

#Merge in JHU data
jhu_cases = pd.read_csv('data/jhu/county_cases_04-30-2020.csv')
jhu_cases = jhu_cases.loc[:, ['Admin2', 'Province_State', 'Confirmed', 'FIPS']]
merged = df_merged.merge(jhu_cases, how='inner', on='FIPS')
merged = merged.drop(['Admin2', 'Province_State'], axis=1)
merged = merged.rename({'Confirmed': 'covid_cases'}, axis=1)

jhu_testing = pd.read_csv('data/jhu/statecovid_04-30-2020.csv')
jhu_testing = jhu_testing[['Province_State', 'Testing_Rate']]
jhu_testing = jhu_testing.rename({'Province_State': 'state_name'}, axis=1)
merged = merged.merge(jhu_testing, how='inner', on='state_name')


#Merge in political data
govparty = pd.read_csv('data/kff_statepoliticalparties.csv', skiprows=2)
govparty = govparty.loc[1:, ['Location', 'Governor Political Affiliation']]
govparty.rename({'Location': 'state_name', 'Governor Political Affiliation': 'gov_party'}, axis=1, inplace=True)
# Convert gov_party to a boolean
govparty['gov_party'] = govparty['gov_party'] = [
    1 if (x == 'Republican') else 0 for x in govparty['gov_party']]
merged = merged.merge(govparty, how='inner', on='state_name')

election = pd.read_csv('data/countypres_2000-2016.csv')
election = election[election['year'] == 2016]
election['prop_votes'] = election['candidatevotes'] / election['totalvotes']
election = election[(election['candidate'] == 'Hillary Clinton') | (election['candidate'] == 'Donald Trump')]

election_grp = election.groupby(['candidate', 'FIPS']).agg({'candidatevotes': 'sum', 'totalvotes': 'sum'}).reset_index()
election_grp['prop_votes'] = election_grp['candidatevotes'] / election_grp['totalvotes']

election_clean = election_grp.pivot(index='FIPS', columns='candidate', values='prop_votes').reset_index()
election_clean['election_diff'] = election_clean['Donald Trump'] - election_clean['Hillary Clinton']
election_clean = election_clean.loc[:, ['FIPS', 'election_diff']]

merged = merged.merge(election_clean, how='inner', on=['FIPS'])


#Unemployment data
abbrev_us_state = dict(map(reversed, US_STATE_ABBREV.items()))

unemp = pd.read_excel('data/laucntycur14.xlsx', skiprows=4)
unemp = unemp[1:-3] # Removing NA rows and notes from original file

unemp['FIPS'] = unemp['LAUS Code'].str.slice(start=2, stop=7).astype('int')
unemp = unemp.loc[:, ['Period', '(%)', 'FIPS']]
unemp_long = unemp.pivot(index='FIPS', columns='Period', values='(%)').reset_index()

unemp_final = unemp_long[['FIPS', 'Apr-19', 'Mar-19', 'Feb-20', 'Mar-20 p']]
# unemp_final['state_name'] = unemp_final['state_abbr'].map(abbrev_us_state)
unemp_final.rename({'Mar-20 p': 'Mar-20'}, axis=1, inplace=True)

merged = merged.merge(unemp_final, how='inner', on='FIPS')

# Read in stay at home policies
policies = pd.read_pickle('data/state_policies.pk1')

state_policies = policies[policies['county_name'].isnull()].drop(columns='county_name')
merged = merged.merge(state_policies, how='left', on='state_name')
merged.rename({'days_closed': 'state_days_closed'}, axis=1, inplace=True)

#Then, merge on county
merged = merged.merge(policies, how='left', on=['county_name', 'state_name'])

#If it didn't merge on county, replace with state value
merged['days_closed'] = np.where(
    merged['days_closed'].isna(), merged['state_days_closed'], merged['days_closed'])
merged.drop(columns='state_days_closed', inplace=True)

merged.to_pickle('data/final_dataset.pk1')
