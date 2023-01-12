import warnings
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

def convert(path=Path('./processedDataset'), out_path=Path('./optimalDataset')):
    data = pd.read_parquet(path / 'train_data.parquet')

    # Process
    categorical_columns = ['B_30', 'B_38', 'D_114', 'D_116', 'D_117', 'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68']
    numeric_columns = list(set(data.columns).difference(categorical_columns + ['customer_ID', 'S_2']))

    map_functions = {}
    for column in train_df.columns:
        if column == 'customer_ID':
            continue
        elif column in categorical_columns:
            map_functions[column] = 'max'
        else:
            map_functions[column] = 'sum'
    groupby_data = data.groupby('customer_ID').agg(map_functions)

    # Build pipepline to process missing data
    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='median')
    # Preprocessing for categorical data
    categorical_transformer = SimpleImputer(strategy='most_frequent')
    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler())
        ]
    )
    # Pipeline
    optimal_data = pipeline.fit_transform(groupby_data)
    optimal_data_df = pd.DataFrame(optimal_data, columns=groupby_data.columns, index=groupby_data.index)

    # save 
    optimal_data_df.to_parquet(out_path / 'data.parquet', index=False)

if __name__ == "__main__":
    convert()
