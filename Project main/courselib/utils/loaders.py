import os
import pandas as pd

def load_or_download_csv(file_name, url, column_names=None, encoding='utf-8'):
    if os.path.exists(file_name):
        print(f"Loading from local `{file_name}`...")
        return pd.read_csv(file_name, index_col=0, encoding=encoding)
    else:
        print(f"Downloading from `{url}`...")
        df = pd.read_csv(url, names=column_names, encoding=encoding)
        df.to_csv(file_name, encoding=encoding)
        print("Saved to local file.")
        return df
    

def load_irises():
    file_name = 'iris_data_set.csv'
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    column_names = ['sepal length [cm]', 'sepal width [cm]',
                'petal length [cm]', 'petal width [cm]', 'iris type']
    
    df = load_or_download_csv(file_name, url, column_names)

    return df