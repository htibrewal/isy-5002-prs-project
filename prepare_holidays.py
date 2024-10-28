import pandas as pd

fixed_holidays = pd.DataFrame([
    {
        'holiday': 'New Year',
        'ds': ['2023-01-01', '2024-01-01'],
        'lower_window': 0,
        'upper_window': 1
    },
    {
        'holiday': 'Labour Day',
        'ds': ['2023-05-01', '2024-05-01']
    },
    {
        'holiday': 'National Day',
        'ds': ['2023-08-09', '2024-08-09'],
        'lower_window': 0,
        'upper_window': 1
    },
    {
        'holiday': 'Christmas Day',
        'ds': ['2023-12-25', '2024-12-25'],
        'lower_window': -1,
        'upper_window': 0
    }
]).explode('ds')

variable_holidays = pd.DataFrame([
    {'holiday': 'Chinese New Year', 'ds': ['2023-01-22', '2024-02-10']},
    {'holiday': 'Chinese New Year Day 2', 'ds': ['2023-01-23', '2024-02-11']},
    {'holiday': 'Good Friday', 'ds': ['2023-04-07', '2024-03-29']},
    {'holiday': 'Hari Raya Puasa', 'ds': ['2023-04-22', '2024-04-10']},
    {'holiday': 'Vesak Day', 'ds': ['2023-06-02', '2024-05-22']},
    {'holiday': 'Hari Raya Haji', 'ds': ['2023-06-29', '2024-06-17']},
    {'holiday': 'Deepawali', 'ds': ['2023-11-12', '2024-10-31']}
]).explode('ds')

sg_holidays = pd.concat((fixed_holidays, variable_holidays))
