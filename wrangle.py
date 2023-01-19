import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import os
from env import get_connection

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer



def get_zillow_data():
    
    '''
    This function is to get the zillow dataset from a local csv file or from SQL Ace to our working notebook to be able to
    use the data and perform various tasks using the data
    ''' 
    
    if os.path.isfile('zillow.csv'):
        
        return pd.read_csv('zillow.csv')
    
    else:
       
        url = get_connection('zillow')
        
        test = '%'
        
        query = '''
        SELECT id, parcelid, bedroomcnt, bathroomcnt,
               calculatedfinishedsquarefeet, fips, 
               yearbuilt, taxamount, taxvaluedollarcnt 
        FROM properties_2017
        JOIN propertylandusetype USING(propertylandusetypeid)
        WHERE propertylandusetypeid = 261;
        ''' 

        df = pd.read_sql(query, url)
        
        df.to_csv('zillow.csv', index = False)

        return df   


    
def rename_columns(df):
    
    df = df.rename(columns={'bedroomcnt':'bedrooms', 
                   'bathroomcnt':'bathrooms',
                   'calculatedfinishedsquarefeet':'sq_ft',
                   'yearbuilt':'year_built',
                   'taxamount':'tax_amount',
                   'taxvaluedollarcnt':'tax_value'})
    return df

def outlier_remove(df):

    df = df[(df.bedrooms <= 6) & (df.bedrooms > 0)]
    
    df = df[(df.bathrooms <= 6) & (df.bathrooms >= 1)]

    df = df[df.tax_value < 2_000_000]

    df = df[df.sq_ft < 10000]

    return df
    
def clean_zillow_data(df):
    
    df = rename_columns(df)
    
    df = df.dropna() 
    
    df = outlier_remove(df)
    
    df.drop_duplicates(inplace=True)

    df.to_csv("zillow.csv", index=False)

    return df    


def train_val_test(df, stratify = None):
    seed = 22
    
    ''' This function is a general function to split our data into our train, validate, and test datasets. We put in a dataframe
    and our target variable to then return us the datasets of train, validate and test.'''
    
    train, test = train_test_split(df, train_size = 0.7, random_state = seed, stratify = None)
    
    validate, test = train_test_split(train, train_size = 0.5, random_state = seed, stratify = None)
    
    return train, validate, test


def wrangle_zillow():
  
    df = get_zillow_data()

    df = clean_zillow_data(df)
    
    return df 



def scaled_data(train, 
               validate, 
               test, 
               columns_to_scale=['bedrooms', 'bathrooms', 'sq_ft', 'tax_amount'],
               return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     make the thing
    mms = MinMaxScaler()
    #     fit the thing
    mms.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(mms.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(mms.transform(validate[columns_to_scale]), 
                                                     columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(mms.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled


# Vizs
                 
                 
def plot_variable_pairs(df):
    sns.pairplot(data = df.sample(2000), kind='reg', diag_kind='hist')
    
    return plt.show()
                 
                 
def plot_categorical_and_continuous_vars(df, cat, cont):
    
    df_sample = df.sample(2000)
    
    sns.barplot(x=cat, y=cont, data=df_sample)
    plt.figure()

    sns.stripplot(x=cat, y=cont, data=df_sample)
    plt.figure()
    
    sns.boxplot(x=cat, y=cont, data=df_sample)
    plt.figure()
    
    
    return plt.show()
                 
                 
                 
                 
    
    