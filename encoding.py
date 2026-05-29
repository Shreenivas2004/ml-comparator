import pandas as pd
import numpy as np
from pandas.core.api import DataFrame
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder



def encoding(x:DataFrame,y:pd.DataFrame,encoder="label"):
    x_encoded = {}
    if type(y.iloc[0]) == str:
        le_y = LabelEncoder() if encoder=="label" else OrdinalEncoder()
        y_encoded = le_y.fit_transform(y)
    else:
        y_encoded = y
    for i in x.columns:
        if type(x.loc[0,i]) == str:
            le_x = LabelEncoder() if encoder=="label" else OrdinalEncoder()
            encoding = le_x.fit_transform(x[i])
            x_encoded[i] = encoding
        else:
            x_encoded[i] = x[i]
    return pd.DataFrame(x_encoded), y_encoded


def encoding_cluster(x):
    x_encoded = {}
    for i in x.columns:
        if type(x.loc[0,i]) == str:
            le_x = LabelEncoder() 
            encoding = le_x.fit_transform(x[i])
            x_encoded[i] = encoding
        else:
            x_encoded[i] = x[i]
    return pd.DataFrame(x_encoded)
