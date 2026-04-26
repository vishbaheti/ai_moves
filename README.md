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
1.  `ukba01b2014.xls`
2.  `ukbaa01b2015.xls`
3.  `ukbaa01b2016.xls`
4.  `ukbusinessworkbook2017.xls`
5.  `ukbusinessworkbook2018.xls`
6.  `ukbusinessworkbook2019.xlsx`
7.  `ukbusinessworkbook2020.xlsx`
8.  `ukbusinessworkbook2021.xlsx`
9.  `ukbusinessworkbook2022.xlsx`
10. `ukbusinessworkbook2023.xlsx`
11. `ukbusinessworkbook2024.xlsx`
12. `ukbusinessworkbook2025new.xlsx`

### Census Data
* `752526794144824.csv` (Raw 2011 Census commuting flows)
