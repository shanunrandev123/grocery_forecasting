import pandas as pd
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb
# from lightgbm import LGBMRegressor
import xgboost as xgb
# from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold




def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'





main_df_train = pd.read_csv(r'C:\Users\Asus\Downloads\train (2).csv')

main_df_test = pd.read_csv(r'C:\Users\Asus\Downloads\test (2).csv')


# print('training dataset--------------------->')
# print(main_df_train.head())


# print('Test dataset ----------------------------->')
# print(main_df_test.head())




base_features = main_df_test.drop(columns=['id'], axis=1).columns

test_id = main_df_test['id']


training_main_df = pd.concat([main_df_train[base_features], main_df_train['orders']], axis=1)

testing_main_df = main_df_test[base_features]


main_df = pd.concat([training_main_df, testing_main_df], axis=0, sort=False).reset_index(drop=True)


print(main_df.tail())



main_df['date'] = pd.to_datetime(main_df['date'])
main_df['month'] = main_df['date'].dt.month
main_df['day'] = main_df['date'].dt.day
main_df['year'] = main_df['date'].dt.year
main_df['day_of_week'] = main_df['date'].dt.dayofweek
main_df['day_of_year'] = main_df['date'].dt.dayofyear
main_df['is_weekend'] = main_df['day_of_week'].apply(lambda x: 1 if x in [5, 6] else 0)
main_df['season'] = main_df['month'].apply(get_season)
main_df['warehouse'] = main_df['warehouse'].astype('category')
main_df['season'] = main_df['season'].astype('category')
main_df['sin_month'] = np.sin(2 * np.pi * main_df['month']/12)
main_df['cos_month'] = np.cos(2 * np.pi * main_df['month']/12)
main_df['sin_day'] = np.sin(2 * np.pi * main_df['day']/30)
main_df['cos_day'] = np.cos(2 * np.pi * main_df['day']/30)
main_df['sin_day_of_week'] = np.sin(2 * np.pi * main_df['day_of_week']/7)
main_df['cos_day_of_week'] = np.cos(2 * np.pi * main_df['day_of_week']/7)
main_df['sin_day_of_year'] = np.sin(2 * np.pi * main_df['day_of_year']/365)
main_df['cos_day_of_year'] = np.cos(2 * np.pi * main_df['day_of_year']/365)


main_df['holiday_name'].fillna('None', inplace=True)


# print('total number of unique holidays in the dataset')
# print(main_df.holiday_name.value_counts())


#encoding the holiday_name column


enc = OneHotEncoder()

holiday_encoded = enc.fit_transform(main_df[['holiday_name']])

encoded_df = pd.DataFrame(holiday_encoded.toarray(), columns=enc.get_feature_names_out(['holiday_name']))

main_df = pd.concat([main_df, encoded_df], axis=1)

main_df.drop(columns=['holiday_name'], axis=1, inplace=True)

# print(main_df.isnull().sum())
# print(main_df.columns)


# print(main_df.info())

main_df['holiday_after'] = main_df['holiday'].shift(-1).fillna(0).astype(int)

main_df['holiday_before'] = main_df['holiday'].shift(1).fillna(0).astype(int)


