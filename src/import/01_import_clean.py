import pandas as pd
import argparse

def load_data(input_filename):
    data = pd.read_csv(input_filename)
    return data

def pivot(dataframe):
    student_tests = dataframe.pivot_table(
        index = 'StudentIdentifier', 
        columns = 'AssessmentName', 
        values = 'ScaleScore', 
        aggfunc = 'max'
    ).reset_index()
    
    data = pd.merge(dataframe, student_tests, on = 'StudentIdentifier')

    data = data.drop(['AssessmentName', 'ScaleScoreAchievementLevel', 'ScaleScore'], axis = 1)
    data = data.drop_duplicates(subset=['StudentIdentifier'])
    return data

def singular_ela_column(dataframe):
    dataframe['ELA Summative'] = (dataframe['ELA Summative Grade 11'].
        combine_first(dataframe['ELA Summative Grade 6']).
        combine_first(dataframe['ELA Summative Grade 7']).
        combine_first(dataframe['ELA Summative Grade 8'])
                            )
    ELA_cols = ['ELA Summative Grade 11',
              'ELA Summative Grade 6',
              'ELA Summative Grade 7',
              'ELA Summative Grade 8']

    dataframe = dataframe.drop(ELA_cols, axis = 1)
    return dataframe

def export(dataframe, output_file):
    dataframe.to_csv(output_file, index = False)

def main(args):
    data = load_data(args.input_filename)
    data = pivot(data)
    data = singular_ela_column(data)
    data = data.dropna(axis=1, how='all')
    
    export(data, args.output_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_filename', type=str, required=True)
    parser.add_argument('--output_filename', type=str, required=True)

    args = parser.parse_args()

    main(args)
    