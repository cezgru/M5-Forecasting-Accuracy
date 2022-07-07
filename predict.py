import numpy as np
import pandas as pd
from downcast import reduce

def create_predictions(directory):
    calendar = pd.read_csv(f'{directory}/calendar.csv')
    sales = pd.read_csv(f'{directory}/sales_train_evaluation.csv')
    sell_prices = pd.read_csv(f'{directory}/sell_prices.csv')
    
    calendar = reduce(calendar)
    sell_prices = reduce(sell_prices)
    sales = reduce(sales)
    sales_d = pd.melt(
        sales_df,
        id_vars=['id','item_id','dept_id','cat_id','store_id','state_id'],
        var_name='d',
        value_name='items_sold').merge(calendar, on='d', how='left').merge(sell_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    sales_d['event_num'] = (sales_d['event_name_1'].notna()).astype(int) + (sales_d['event_name_2'].notna()).astype(int)
    sales_d.loc[sales_d['state_id'] == 'CA', 'snap'] = sales_d.loc[sales_d['state_id'] == 'CA']['snap_CA']
    sales_d.loc[sales_d['state_id'] == 'TX', 'snap'] = sales_d.loc[sales_d['state_id'] == 'TX']['snap_TX']
    sales_d.loc[sales_d['state_id'] == 'WI', 'snap'] = sales_d.loc[sales_d['state_id'] == 'WI']['snap_WI']
    sales_d['snap'] = sales_d['snap'].astype(int)
    sales_d['weekend'] = (sales_d['wday'] <= 2).astype(int)
    sales_d['season'] = (sales_d['month'] / 3).astype(int)
    sales_d['season'] = (sales_d['season'] % 4).astype(int)
    c_features = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

    for c in c_features:
        le = LabelEncoder()
        sales_d[c+'_label'] = le.fit_transform(sales_d[c])
    lags = [7, 28, 35, 42, 60, 360]
    lag_columns = [f'lag_{lag}' for lag in lags] # name columns
    for lag, lag_column in zip(lags, lag_columns):
        sales_d[lag_column] = sales_d[['id',"items_sold"]].groupby('id')['items_sold'].shift(lag)
    rolls = [7, 28, 35, 42, 60, 360]
    for r in rolls:
        for lag, lag_column in zip(lags, lag_columns):
            sales_d[f'rmean_{lag}_{r}'] = sales_d[['id', lag_column]].groupby('id')[lag_column].transform(lambda x: x.rolling(r).mean())
    unused = ['index', 'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id','wm_yr_wk','items_sold',]
    features = list(set(train.columns)-set(unused))
    
    X = sales_d[features]
    
    filename = 'lgbm_final.sav'
    pickle.dump(lgb_reg, open(filename, 'wb'))
    
    return lgb_reg.predict(X)

    if __name__ == "__main__":
        print('This script doesnt make too much sense. The entire premise is to deploy a very specific competition submission.')
        print('Nevertheless, this script will provide you with an array containing item sale predictions. Use it as you will.\n')
        print('\nWord of warning: this program may fail due to memory constraints if the dataset is too large.')
        print('As such, we recommend only using a small dateset for future predictions. Its not as if you will use it to predict years in advance.\n')
        print('\nFor clarification:')
        print('\nInput: contains calendar sales and prices shops etc. for n days')
        print('\nOutput: predicts item sales for n days\n')
        directory = input('\nPlease provide the path to the directory containing calendar.csv, sales_train_evaluation.csv, sell_prices.csv')
        print(create_predictions(directory))
    
