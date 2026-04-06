import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

def load_data(input_filename):
    data = pd.read_csv(input_filename)
    return data

def data_split(dataframe, target_col, test_size=0.2, random_state=42):
    X = dataframe.drop(columns=[target_col])
    y = dataframe[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = test_size,
        random_state = random_state
    )

    train_df = X_train.copy()
    train_df[target_col] = y_train

    test_df = X_test.copy()
    test_df[target_col] = y_test

    return train_df, test_df

def export(train_df, val_df, test_df, train_output, val_output, test_output):
    train_df.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)
    test_df.to_csv(test_output, index=False)
    
def main(args):
    data = load_data(args.input_filename)
    train_df, test_df = data_split(data, "ELA Summative")
    val_df, test_df = data_split(test_df, "ELA Summative", test_size = 0.5)
    export(train_df, val_df, test_df, args.train_output, args.val_output, args.test_output)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_filename", type=str, required=True)
    parser.add_argument("--train_output", type=str, required=True)
    parser.add_argument("--val_output", type=str, required=True)
    parser.add_argument("--test_output", type=str, required=True)

    args = parser.parse_args()
    
    main(args)
    