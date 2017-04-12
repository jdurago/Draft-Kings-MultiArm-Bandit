import pandas as pd


def import_draftkings_salaries(path_to_filename):
    df = pd.read_csv(path_to_filename)
    df['DollarPerPoint'] = df['Salary'] / df['AvgPointsPerGame']

    df = df.sort_values(['AvgPointsPerGame', 'Salary', 'Position'], ascending=[0, 0, 0])
    return df


def get_projections(path_to_filename):
    df = pd.read_pickle(path_to_filename)

    return df