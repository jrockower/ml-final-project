# Unemployment trends amidst COVID-19 and its governance

## How to run analysis

1. [`stay_home_dates.py`](./stay_home_dates.py)
   * Reads in and cleans information on stay at home orders from counties and states.
   * Saves to .pk1 files.
2. [`data_merge.py`](./data_merge.py)
   * Cleans data and builds features for machine learning models.
3. [`analysis.ipynb`](./analysis.ipynb)
   * Runs various machine learning models.

## Assorted Programs

1. [`pipeline.py`](./pipeline.py)
   * Helper functions to be used in [`analysis.ipynb`](./analysis.ipynb).
2. [`visualizations.ipynb`](./visualizations.ipynb)
   * Jupyter Notebook to build various visualizations to be used in report.

## Data Folder

1. acs
   * County-level unemployment data through March and April.
2. elections
   * Data on county-level presidential election margins.
   * State government political party data.
3. interventions
   * COVID-19 policies at the state- and (when available) county-level.
4. jhu
   * COVID-19 cases from Johns Hopkins University.
5. pickle
   * Directory containing cleaned Pandas dataframes.

