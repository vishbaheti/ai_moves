## File Descriptions

* **build_panel.ipynb**: Merges 12 years of ONS business data (2014–2025) into one file.
* **build_commute_matrix.py**: Processes raw 2011 Census data to create a commuting map.
* **compute_exposure.py**: Calculates the AI exposure index for every district.
* **validate_index.ipynb**: Runs robustness checks (Monte Carlo noise simulations).
* **build_figure.py**: Creates the main analysis charts.
* **dashboard.py**: Runs an interactive Streamlit dashboard.
* **data_exploration.ipynb**: Basic exploration of data

## Required Data

### ONS Business Data
Place these files in your data folder:
* `ukba01b2014.xls` through `ukbusinessworkbook2025new.xlsx`

### Census Data
* `752526794144824.csv` (Raw 2011 Census commuting flows)
