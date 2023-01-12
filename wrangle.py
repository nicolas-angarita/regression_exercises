import pandas as pd
import numpy as np

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
                   'calculatedfinishedsquarefeet':'sqft',
                   'yearbuilt':'year_built',
                   'taxamount':'tax_amount',
                   'taxvaluedollarcnt':'tax_value'})
    return df

def clean_nulls(df):    

    df = df.dropna()
    
    return df
    
def clean_zillow_data(df):
    
    df = rename_columns(df)
    
    df = clean_nulls(df)    

    df.to_csv("zillow.csv", index=False)

    return df    


def wrangle_zillow():

    file_name = "zillow.csv"

    if os.path.isfile(file_name):
        df = pd.read_csv(file_name)
    else:
        df = get_zillow_data()

        df = clean_zillow_data(df)

    return df



def train_val_test(df,col):
    seed = 22
    
    ''' This function is a general function to split our data into our train, validate, and test datasets. We put in a dataframe
    and our target variable to then return us the datasets of train, validate and test.'''
    
    train, test = train_test_split(df, train_size = 0.7, random_state = seed, stratify = df[col])
    
    validate, test = train_test_split(train, train_size = 0.5, random_state = seed, stratify = train[col])
    
    return train, validate, test    
    