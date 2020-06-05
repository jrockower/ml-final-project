import pandas as pd
import numpy as np
from datetime import date

#Read in stay at home order dates and fix some typos
sah_df = pd.read_excel('data/interventions/stay_at_home_orders.xlsx', sheet_name='Clean', header=0)
sah_df.loc[2, 'County'] = 'Anchorage'
sah_df['County'] = sah_df['County'].replace('all', np.NaN)

#Separate into state and county level
sah_county = sah_df.dropna(subset=['County'])
sah_state = sah_df[sah_df.loc[:, 'County'].isna()].iloc[:, [0, 3]]

#Read in state-level policy dataset
state_orders = pd.read_csv('data/interventions/covid_state_policy.csv')
state_closed = state_orders.loc[:50, ['State', 'Closed non-essential businesses']]

#Merge together state data
state_closed = state_closed.merge(sah_state, on='State', how='outer')
state_closed.rename({'Closed non-essential businesses': 'bus_close', 'Date': 'sah_date'}, axis=1, inplace=True)


#Replace some missing values (replacing with SAH date makes later steps easier)
state_closed['bus_close'].where(state_closed['bus_close'] != '0', state_closed['sah_date'], inplace=True)
state_closed['bus_close'] = pd.to_datetime(state_closed['bus_close'], infer_datetime_format=True)

#Take the earlier date
state_closed['state_date'] = np.where(state_closed['sah_date'] < state_closed['bus_close'],
                                      state_closed['sah_date'], state_closed['bus_close'])

#Calculate days under order
period_end = date(2020, 4, 20)
state_closed['end_date'] = period_end
state_closed['end_date'] = pd.to_datetime(state_closed['end_date'], infer_datetime_format=True)
state_closed['days_closed_state'] = (state_closed['end_date'] - state_closed['state_date']).dt.days
state_closed['days_closed_state'].fillna(0, inplace=True)

#Do the same for county
sah_county = sah_county.iloc[:, [0, 1, 3]]
sah_county['end_date'] = period_end
sah_county['end_date'] = pd.to_datetime(sah_county['end_date'], infer_datetime_format=True)
sah_county['days_closed_county'] = (sah_county['end_date'] - sah_county['Date']).dt.days
sah_county['days_closed_county'].fillna(0, inplace=True)


#Merge these together, because sometimes state days closed is greater than county days
sah_county_state = sah_county.merge(state_closed, how='left', on='State')
sah_county_state['days_closed'] = np.where(sah_county_state['days_closed_county'] > sah_county_state['days_closed_state'],
                                sah_county_state['days_closed_county'], sah_county_state['days_closed_state'])

county_policy_df = sah_county_state.loc[:, ['State', 'County', 'days_closed']]
county_policy_df.rename({'State': 'state_name', 'County': 'county_name'}, axis=1, inplace=True)

state_policy_df = state_closed.loc[:, ['State', 'days_closed_state']]
state_policy_df.rename({'State': 'state_name'}, axis=1, inplace=True)

county_policy_df.to_pickle('data/pickle/county_policies.pk1')
state_policy_df.to_pickle('data/pickle/state_policies.pk1')

