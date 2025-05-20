import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess():
    # Get the directory where preprocess.py is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define relative paths to your three data files
    data_files = {
        'Stand': os.path.normpath(os.path.join(script_dir, '..', 'data', 'accelerometer_stand.txt')),
        'Walk': os.path.normpath(os.path.join(script_dir, '..', 'data', 'accelerometer_walk.txt')),
        'Run': os.path.normpath(os.path.join(script_dir, '..', 'data', 'accelerometer_run.txt')),
    }

    # Example: Load each file into a DataFrame or process as needed
    dfStand = pd.read_csv(data_files['Stand'], sep='\t', header=None, names=['x', 'y', 'z'], encoding='utf-16')
    dfWalk = pd.read_csv(data_files['Walk'], sep='\t', header=None, names=['x', 'y', 'z'], encoding='utf-16')
    dfRun = pd.read_csv(data_files['Run'], sep='\t', header=None, names=['x', 'y', 'z'], encoding='utf-16')
    
    # Add class labels
    dfStand['class'] = 'Stand'
    dfWalk['class'] = 'Walk'
    dfRun['class'] = 'Run'
    
    # Concatenate all dataframes
    df = pd.concat([dfStand, dfWalk, dfRun], ignore_index=True)

    # Perform stratified train-test split
    X = df[['x', 'y', 'z']]  # Features
    y = df['class']          # Labels
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=63, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


    