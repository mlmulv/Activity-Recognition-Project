import pandas as pd
from sklearn.model_selection import train_test_split
import os
import glob

def preprocess(opt = True, windowSize = 5, testSize = 0.25):
    """
    Inputs:
    opt - if true you want the data to be filtered by moving average (default is true)
        windowSize - size of the window for the moving average (default is 5)
        testSize - the percentage of samples dedicated to the test group (default is 0.25)

    Outputs:
    X_train - x,y,z accelerometer data for train set
    X_test - x,y,z accelerometer data for test set
    y_train - class labels for train set
    y_test - class labels for test set 
    """
    
    # Define the base directory for data relative to the script location
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    
    # Define activity labels and their corresponding directories
    activity_dirs = {
        'run': os.path.join(base_dir, 'run'),
        'walk': os.path.join(base_dir, 'walk'),
        'stand': os.path.join(base_dir, 'stand')
    }
    
    # Initialize a list to hold all data
    data = []

    for activity, dir_path in activity_dirs.items():
        file_list = glob.glob(os.path.join(dir_path, '*.csv'))
        for file_path in file_list:
            # Load each CSV into a dataframe
            df = pd.read_csv(file_path)

            # Add a class label column
            df['class'] = activity
            
            # Append to the list
            data.append(df)
    
    # Combine all dataframes into one dataframe
    df = pd.concat(data, ignore_index=True)

    # Rename columns to desired feature names
    df.rename(columns={
        'userAcceleration.x': 'xAcc',
        'userAcceleration.y': 'yAcc',
        'userAcceleration.z': 'zAcc'
    }, inplace=True)

    # Perform moving average if specified
    if opt:
        df[['xAcc', 'yAcc', 'zAcc']] = df[['xAcc', 'yAcc', 'zAcc']].rolling(window=windowSize).mean()

    # Converting to integers for classes
    label_mapping = {
        'stand': 0,
        'walk': 1,
        'run': 2
    }
    
    df['class'] = df['class'].map(label_mapping)
    
    # Remove nan rows if necessary
    df = df.dropna()
    
    # Perform stratified train-test split
    X = df[['xAcc', 'yAcc', 'zAcc']]  # Features
    y = df['class']          # Labels

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=testSize, random_state=63, stratify=y)
        
    return X_train, X_test, y_train, y_test