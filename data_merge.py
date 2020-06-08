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
B03001_003E - hispanic pop

B28002_013E - no internet
B28011_001E - internet total

B15003_001E - education total
B15003_022E - bachelor's degree
B15003_023E - master's degree
B15003_024E - professional degree
B15003_025E - doctorate degree

B19013_001E - median family income

C24050_001E - industry by occupation total
C24050_012E - arts, entertainment, recreation, accomodation, food services

'''

import requests
import pandas as pd
import numpy as np



#Import and clean ACS and Census data
HOST = "https://api.census.gov/data"
year = "2018"
dataset = "acs/acs5"
base_url = "/".join([HOST, year, dataset])
predicates = {}
get_vars = ['NAME', 'B02001_001E', 'B02001_002E', 'B02001_003E', 'B03001_003E',
            'B28002_013E', 'B28011_001E', 'B15003_001E', 'B15003_022E',
            'B15003_023E', 'B15003_024E', 'B15003_025E',
            'B19013_001E', 'C24050_001E', 'C24050_012E']
predicates["get"] = ",".join(get_vars)
predicates["for"] = "county:*"
r = requests.get(base_url, params=predicates)

df = pd.DataFrame(data=r.json())
df.columns = df.iloc[0]
df = df[1:]

#Separate county and state name
df['county_name'] = df['NAME'].str.extract('(.*) \w*,')
df['state_name'] = df['NAME'].str.extract(', (.*)')


#Clean columns and column names
#Change column types to int
cols_to_change = ['B02001_001E', 'B02001_002E', 'B02001_003E', 'B03001_003E',
            'B28002_013E', 'B28011_001E', 'B15003_001E', 'B15003_022E',
            'B15003_023E', 'B15003_024E', 'B15003_025E',
            'B19013_001E', 'C24050_001E', 'C24050_012E']
for col in cols_to_change:
    df[col] = pd.to_numeric(df[col])

#Race
df.rename({'B02001_001E': 'total_pop', 'B19013_001E': 'median_income'}, axis=1, inplace=True)
df['prop_white'] = df['B02001_002E'] / df['total_pop']
df['prop_black'] = df['B02001_003E'] / df['total_pop']
df['prop_hisp'] = df['B03001_003E'] / df['total_pop']

#Income
df['log_med_income'] = np.log(df['median_income'])

#Internet access
df['prop_no_internet'] = df['B28002_013E'] / df['B28011_001E']

#Education
df['prop_ba'] = (df['B15003_022E'] + df['B15003_023E'] + df['B15003_024E'] +
                 df['B15003_025E']) / df['B15003_001E']

#Occupation
df['prop_services'] = df['C24050_012E'] / df['C24050_001E']

#Drop extra columns
df_clean = df.drop(['B02001_002E', 'B02001_003E', 'B03001_003E',
            'B28002_013E', 'B28011_001E', 'B15003_022E', 'B15003_023E',
            'B15003_024E', 'B15003_025E', 'B15003_001E', 'C24050_001E',
            'C24050_012E'], axis=1)


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
govparty = pd.read_csv('data/elections/kff_statepoliticalparties.csv', skiprows=2)
govparty = govparty.loc[1:, ['Location', 'Governor Political Affiliation']]
govparty.rename({'Location': 'state_name', 'Governor Political Affiliation': 'gov_party'}, axis=1, inplace=True)

# Convert gov_party to a boolean
govparty['gov_party'] = govparty['gov_party'] = [
    1 if (x == 'Republican') else 0 for x in govparty['gov_party']]
merged = merged.merge(govparty, how='inner', on='state_name')

election = pd.read_csv('data/elections/countypres_2000-2016.csv')
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
unemp = pd.read_excel('data/bls/laucntycur14_april.xlsx', skiprows=4)
unemp = unemp[1:-3] # Removing NA rows and notes from original file

unemp['FIPS'] = unemp['LAUS Code'].str.slice(start=2, stop=7).astype('int')
unemp = unemp.loc[:, ['Period', '(%)', 'FIPS']]
unemp_long = unemp.pivot(index='FIPS', columns='Period', values='(%)').reset_index()

unemp_final = unemp_long[['FIPS', 'Apr-19', 'Mar-19', 'Feb-20', 'Mar-20', 'Apr-20 p']]
unemp_final.rename({'Apr-20 p': 'Apr-20'}, axis=1, inplace=True)

merged = merged.merge(unemp_final, how='inner', on='FIPS')

# Read in stay at home policies
state_policies = pd.read_pickle('data/pickle/state_policies.pk1')
county_policies = pd.read_pickle('data/pickle/county_policies.pk1')

merged = merged.merge(state_policies, how='left', on='state_name')

#Then, merge on county
merged = merged.merge(county_policies, how='left', on=['county_name', 'state_name'])

#If it didn't merge on county, replace with state value
merged['days_closed'] = np.where(
    merged['days_closed'].isna(), merged['days_closed_state'], merged['days_closed'])

merged.drop(columns='days_closed_state', inplace=True)

# Convert unemployment columns to numeric
merged[['pop_density', 'Mar-19', 'Apr-19', 'Feb-20', 'Mar-20',
        'Apr-20']] = merged[['pop_density', 'Mar-19', 'Apr-19',
        'Feb-20', 'Mar-20', 'Apr-20']].apply(pd.to_numeric)

# Add yearly change fields
merged['yearly_change'] = (merged['Apr-20'] - merged['Apr-19']) / merged['Apr-19']
merged['monthly_change'] = (merged['Apr-20'] - merged['Feb-20']) / merged['Feb-20']

merged.to_pickle('data/pickle/final_dataset.pk1')
