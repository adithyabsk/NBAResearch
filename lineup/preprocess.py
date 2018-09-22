#!/usr/bin/env python

import h5py
from functools import reduce

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

import sklearn
from sklearn_pandas import DataFrameMapper, gen_features

def set_pandas_options(max_columns=None, max_rows=None):
    pd.set_option("display.max_columns", max_columns)
    pd.set_option("display.max_rows", max_rows)

def load_live_data():
    gt_live_df = pd.read_hdf('../data/output/LiveTeamBoxScores.h5', 'gt_df')
    gp_live_df = pd.read_hdf('../data/output/LivePlayerBoxScores.h5', 'gt_df')

    gt_live_df.to_csv('../data/tmp/{}gt.csv'.format(year), index=0)
    gp_live_df.to_csv('../data/tmp/{}gp.csv'.format(year), index=0)


def create_rnn_features(indicies):
    pass

def preprocess(time_split=1200):
    # gt_live_df = pd.read_csv('../data/tmp/41700404gt.csv')
    # gp_live_df = pd.read_csv('../data/tmp/41700404gp.csv')

    print('Loading data...')
    gt_live_df = pd.read_hdf('../data/output/LiveTeamBoxScores1617.h5', 'gt_df')
    gp_live_df = pd.read_hdf('../data/output/LivePlayerBoxScores1617.h5', 'gt_df')
    game_info_df = pd.read_csv('../data/Hackathon_Game_Mapping.csv', compression='gzip')
    game_info_df = game_info_df[['Game_id', 'Home_Team_id', 'Visitor_Team_id']]

    def order_columns(df):
        g_id, *_ = df['Game_id']
        home, away = game_info_df.loc[game_info_df['Game_id'] == g_id, ['Home_Team_id', 'Visitor_Team_id']].values[0]
        team1, team2 = df['Team_id']
        df_np = df.values
        if team1 == home:
            return pd.Series(df_np.ravel())
        else:
            return pd.Series(np.flipud(df_np).ravel())


    q_time = 7200
    bins = list(range(q_time, -1, -time_split))
    diff_cols = ['DIFF{}'.format(b) for b in bins]
    cb_zip = zip(diff_cols, bins)

    print('Drop duplicates of player games...')
    filter_df = gp_live_df[['Game_id', 'Period', 'PC_Time', 'Event_Num']].drop_duplicates(keep='last')
    for c, b in cb_zip:
        filter_df[c] = (filter_df['PC_Time'] - b).abs()
    print('Groupby player data...')
    gbpy_gp = filter_df.groupby(['Game_id', 'Period'])
    print('Find closest index to selected times...')
    selector = reduce(lambda x, y: x | y, [set(gbpy_gp[c].idxmin().unique()) for c in diff_cols])
    print('Selector for raw data...')
    selector_df = filter_df.loc[selector, ['Game_id', 'Period', 'PC_Time', 'Event_Num']]
    print('Merging indicies to raw data...')
    gt_live_df = pd.merge(gt_live_df, selector_df, on=['Game_id', 'Period', 'PC_Time', 'Event_Num'])
    gp_live_df = pd.merge(gp_live_df, selector_df, on=['Game_id', 'Period', 'PC_Time', 'Event_Num'])


    active_cols = ["Active{}".format(i) for i in range(10)]
    default_cols = ['Player_id', 'Game_id', 'Period', 'PC_Time', 'WC_Time', 'Event_Num']
    gp_orig_cols = gp_live_df.columns
    print('Merging each active player col into team data...')
    for c in tqdm(active_cols):
        gp_live_df.columns = gp_orig_cols.map(lambda x: str(x) + '_{}'.format(c))
        right_on = ['{}_{}'.format(d, c) for d in default_cols]
        gt_live_df =  pd.merge(gt_live_df, 
                               gp_live_df, 
                               how='left',
                               left_on=[c, 'Game_id', 'Period', 'PC_Time', 'WC_Time', 'Event_Num'],
                               right_on=right_on
                               ).drop(columns=[c, 'Team_id_{}'.format(c), '{}_{}'.format(c, c)]+right_on)
    print('Dropping plus minus...')
    gt_live_df = gt_live_df.drop(columns=['+/-'])
    print('Dropping extra active cols...')
    gt_live_df = gt_live_df[gt_live_df.columns.drop(list(
        gt_live_df.filter(regex='Active[0-9]_Active[0-9]')))]

    gt_cols = gt_live_df.columns
    final_cols = ['Game_id', 'Period', 'PC_Time', 'WC_Time', 'Event_Num']+['{}_{}'.format(c, t) for t in ['home', 'away'] for c in gt_cols]
    print('Reorder the columns using groupby...')
    gt_live_df = gt_live_df.groupby(['Game_id', 'Period', 'PC_Time', 'WC_Time', 'Event_Num']).progress_apply(order_columns)
    gt_live_df = gt_live_df.reset_index()
    gt_live_df.columns = final_cols

    print('Sort gameplay value data...')
    gt_live_df = gt_live_df.sort_values(
        by=["Period", "PC_Time", "WC_Time", "Event_Num"],
        ascending=[True, False, True, True],
    )

    print('Drop unncessary columns...')
    gt_live_df.drop(columns=['Game_id_home', 'Event_Num_home', 'Period_home',
                             'PC_Time_home', 'WC_Time_home', 'Game_id_away', 
                             'Event_Num_away', 'Period_away', 'PC_Time_away',
                             'WC_Time_away'])
    print(gt_live_df.dtypes)

    print('Computing response variables...')
    scores_df = gt_live_df.loc[:, ['PTS_home', 'PTS_away']]   
    scores_df['RESPONSE'] = ((scores_df['PTS_home'].diff() - scores_df['PTS_away'].diff()) > 0).astype(int)

    return gt_live_df, scores_df['RESPONSE']

def train_test_split(data, pct_test):
    """Splits data into a training and testing sections
    
    Args:
        data: a numpy array of rows of features
        n_test: the amount of data to be tested on (selected from the end of the 
            dataset)
        var_name: the name of the variable to be train test split
        input: whether or not the data is an input var
        timeseries: whether or not the input data is timeseries data (in order to 
            determine whether or not to reshape the data)
    Returns:
        A tuple to be unpacked of training (train_var) and testing (test_var)
    """
    offset = data.shape[0] - int(data.shape[0]*pct_test)

    train_var = data[:offset, :] if data.ndim > 1 else data[:offset]
    test_var = data[offset:, :] if data.ndim > 1 else data[offset:]

    return train_var, test_var

def scale_data(data):
    feature_def = gen_features(
        columns=data.columns.values.reshape((-1, 1)).tolist(),
        classes=[sklearn.preprocessing.StandardScaler]
    )
    mapper = DataFrameMapper(feature_def, 
        df_out=True)

    transformed_data = mapper.fit_transform(data)

    return mapper, transformed_data


def series_to_supervised(x_data, n_prev=7):
    """Converts time series data into a supervised learning problem
    Args:
        x_data: a list of input features per time step
        n_prev: the number of previosu x matches per time step (t-n, ...t-1)
    Returns:
        A tuple to be unpacked of the aggregate of x inputs (X_agg) of 
        type numpy array and the pass through y_inputs modified to be 
        numpy array (y_data)
    """

    print('Creating supervized data from match series')
    X_agg = []
    _, zero_len = x_data.shape
    zeros = np.zeros(zero_len)
    for i, row in tqdm(x_data.iterrows(), total=len(x_data)):
        temporal_arr = []
        for j in range(n_prev, 0, -1):
            if i-n_prev < 0:
                temporal_arr.append(zeros)
            else:
                temporal_arr.append(x_data.iloc[i-n_prev])
        X_agg.append(np.array(temporal_arr))

    X_agg = np.array(X_agg)

    return X_agg

if __name__ == '__main__':
    gt_live_df, resp = preprocess()
    gt_live_df.drop(columns=['Game_id', 'WC_Time', 'Event_Num', 'Team_id_home', 'Team_id_away'])
    mapper, gt_live_df = scale_data(gt_live_df)
    supervised_arr = series_to_supervised(gt_live_df)

    train_X, test_X = train_test_split(supervised_arr, 0.2)
    train_y, test_y = train_test_split(resp.values, 0.2)
    train_y, test_y = train_y.reshape((-1, 1)), test_y.reshape((-1, 1))

    with h5py.File('../data/model_data.h5', 'w') as h5f:
        h5f.create_dataset('train_X', data=train_X)
        h5f.create_dataset('test_X', data=test_X)
        h5f.create_dataset('train_y', data=train_y)
        h5f.create_dataset('test_y', data=test_y)
