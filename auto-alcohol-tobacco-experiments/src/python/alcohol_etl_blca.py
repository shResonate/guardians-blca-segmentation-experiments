

import argparse
import os
import subprocess
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
#from sklearn.preprocessing import LabelEncoder
#from datetime import datetime, timezone
#import miceforest as mf
#import matplotlib.pyplot as plt
import s3fs

pd.set_option('display.max_columns', None)

def main():
    """
    Main function to orchestrate the entire data processing, segmentation, and model persistence workflow.
    """
    fs = s3fs.S3FileSystem(anon=False)
    alcohol_data = read_data_from_s3(fs, args.alcohol_file_path, 251)
    alcohol_data = alcohol_data.drop_duplicates(subset='RID')
    newX = split_bottleneck(alcohol_data)
    pca = get_pca(newX, 12)
    pca_data = apply_pca_with_nulls(newX, pca)
    newX = pca_data.copy()
    newX.to_csv(args.s3_output_directory + 'alcohol_pca_data.csv', index=False)
    # Recode Resonate attributes with the help of the attribute taxonomy
    # Load the attribute taxonomy
    # attribute_taxonomy = pd.read_csv(args.local_directory + 'attribute_taxonomy.csv')
    attributes = attributes_to_recode(newX)
    recoded_data = recode_attributes(newX, attributes)
    save_data_to_csv(recoded_data, args.s3_output_directory, 'alcohol_blca_input.csv')
    save_data_to_csv(recoded_data, args.local_directory, 'alcohol_blca_input.csv')
    inputFilePath = args.local_directory + 'alcohol_blca_input.csv'
    outputPath = args.local_directory
    colsToDrop = 'RID'
    arguments = [inputFilePath, outputPath, args.numSegments, args.burnIn, args.thin, args.iterations, colsToDrop, args.useCase]
    blcaRcodePath = '/Users/samhawala/Documents/GitHub/guardians-segmentation/blcaExperiments/alcoholExperiment/src/R/blca.R'
    os.system('Rscript ' + blcaRcodePath + ' ' + ' '.join(arguments))
    # sync to s3
    print('Syncing to s3...')
    os.system('aws s3 sync ' + output_dir + ' ' + args.s3_output_directory)
########################################################################################################
########################################################################################################

parser = argparse.ArgumentParser(description='Experiment ALCOHOL Segmentation')

parser.add_argument('--useCase', type=str,
                    default="alcohol",
                    help='The current use-case for segmentation')

parser.add_argument('--alcohol_file_path', type=str,
                    default='s3://datasci-scantle/disruptCRM/bsm/alcohol/',
                    help='The location of the AUTO files')

parser.add_argument('--numSegments', type=str,
                    default='5',
                    help='The default number of desired segments')

parser.add_argument('--burnIn', type=str,
                    default='100',
                    help='The number of initial iterations to discard as burn-in')

parser.add_argument('--thin', type=str,
                    default= '0.5',
                    help='The thinning rate to achieve good mixing')

parser.add_argument('--iterations', type=str,
                    default= '1000',
                    help='The number of iterations to run the gibbs sampler for after burn-in')

parser.add_argument('--local_directory', type=str,
                    default='/Users/samhawala/Documents/work2024/Segmentations/finalExperiments/alcoholExperiment/',
                    help='The location of the attribute hierarchy file')

parser.add_argument('--s3_output_directory', type=str,
                    default='s3://resonate-datasci-dev/shawala/Segmentations/alcoholExperiment/',
                    help='the location to save the preprocessed data')

args = parser.parse_known_args()[0]


def read_data_from_s3(fs, file_path, nfiles):
    """
    Read data from S3 the TU data parquet file.

    Args:
        fs: S3FileSystem instance.
        file_path: Path to the S3 directory containing parquet files.
        nfiles: Number of parquet files to read.

    Returns:
        data: Concatenated DataFrame of all parquet files.
    """
    files = sorted(fs.glob(f"{file_path}*.parquet"))[:nfiles]

    if len(files) == 1:  # If only one file is found, read it directly
        try:
            return pd.read_parquet(fs.open(files[0], mode='rb')).drop_duplicates(subset='RID')
        except Exception as e:
            print(f"Error reading file {files[0]}: {e}")
            return pd.DataFrame()  # Return an empty DataFrame if there's an error

    # If multiple files are found, concatenate them
    try:
        data = [pd.read_parquet(fs.open(file, mode='rb')) for file in files]
    except Exception as e:
        print(f"Error reading one or more files: {e}")
        return pd.DataFrame()

    if len(data) > 0:  # Check if there's at least one DataFrame to concatenate
        return pd.concat(data, ignore_index=True).drop_duplicates(subset='RID')
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no files were read
def read_remaining_data_from_s3(fs, remaining_files):
    """
    Read data from S3 for the specified parquet files that were not part of the sample.

    Args:
        fs: S3FileSystem instance.
        remaining_files: List of parquet file paths to read.

    Returns:
        data: Concatenated DataFrame of all remaining parquet files.
    """
    if len(remaining_files) == 0:
        print("No remaining files to read.")
        return pd.DataFrame()  # Return an empty DataFrame if there are no remaining files

    if len(remaining_files) == 1:  # If only one file is found, read it directly
        try:
            return pd.read_parquet(fs.open(remaining_files[0], mode='rb')).drop_duplicates(subset='RID')
        except Exception as e:
            print(f"Error reading file {remaining_files[0]}: {e}")
            return pd.DataFrame()  # Return an empty DataFrame if there's an error

    # If multiple files are found, concatenate them
    data = []
    try:
        for file in remaining_files:
            data.append(pd.read_parquet(fs.open(file, mode='rb')))
    except Exception as e:
        print(f"Error reading one or more files: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

    if len(data) > 0:  # Check if there's at least one DataFrame to concatenate
        return pd.concat(data, ignore_index=True).drop_duplicates(subset='RID')
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no files were read

def missingness_report(df):
    """
    Get the missingness report
    :param df:
    :return missingness:
    """
    missingness = {}
    totalRows = len(df)
    for colName in df.columns:
        missingCount = df[colName].isnull().sum()
        missingPercentage = (missingCount / totalRows) * 100
        missingness[colName] = {"missingCount": missingCount, "missingPercentage": missingPercentage}
    return missingness
def print_missingness(missingness):
    for colName, stats in missingness.items():
        if stats["missingCount"] > 0:
            print(f"{colName}: {stats}")
def split_bottleneck(df):
    """
    Create the PCA data
    :param df:
    :return:
    """
    new_columns = pd.DataFrame(df['bottleneck'].apply(lambda x: pd.Series(x)), dtype=float)
    new_columns.columns = [f'bottleneck_{i}' for i in range(len(new_columns.columns))]
    newX = pd.concat([df, new_columns], axis=1)
    newX.drop(columns=['bottleneck','behavior_type'], inplace=True)
    return newX
def get_pca(df, n_components):
    """
    Get the PCA components
    :param df:
    :param n_components:
    :return:
    """
    bottleneck_cols = [x for x in df.columns if 'bottleneck' in x]
    pca = PCA(n_components=n_components)
    pca.fit(df[bottleneck_cols].dropna())
    return pca
def apply_pca_with_nulls(df, pca):
    """
    Apply the PCA to data, retaining NaN rows as NaN in the output.
    :param df: DataFrame with the original data.
    :param pca: Pre-fitted PCA object.
    :return: DataFrame containing PCA-transformed features or NaNs.
    """
    bottleneck_cols = [x for x in df.columns if 'bottleneck' in x]
    other_cols = [x for x in df.columns if x not in bottleneck_cols]
    # Initialize a DataFrame for PCA output with the same index as df
    bottleneck_pca = pd.DataFrame(index=df.index, columns=[f'pca_{i}' for i in range(pca.n_components_)], dtype=float)
    # Apply PCA only to rows without NaNs
    clean_index = df.dropna(subset=bottleneck_cols).index
    clean_data = df.loc[clean_index, bottleneck_cols]
    transformed_data = pca.transform(clean_data).astype(float)
    # Insert the transformed data back to the corresponding rows
    bottleneck_pca.loc[clean_index] = transformed_data
    result_df = pd.concat([df[other_cols], bottleneck_pca], axis=1)
    return result_df
def attributes_to_recode(df):
    """
    Get the attributes to recode
    :param df:
    :return:
    """
    # Get the columns that contain the attribute values
    attribute_cols = [col for col in df.columns if re.search(r'^\d+$', col)]
    # Get the unique values of the attribute columns
    unique_values = df[attribute_cols].stack().unique()
    # Create a DataFrame to store the attribute taxonomy
    attribute_taxonomy = pd.DataFrame(unique_values, columns=['AttributeID'], dtype=int)
    exclude_cols = ['RID', 'bottleneck']
    # # Identify columns that have values not equal to 0 or 1
    cols_not_01 = [col for col in df.columns if col not in exclude_cols and
                   df[col].isin([0, 1]).all() == False]
    # Output the column names that contain values other than 0 or 1
    return cols_not_01
def recode_attributes(df, attributes):
    """
    Recode the attributes based on the attribute taxonomy.
    :param df: DataFrame containing the original attributes.
    :param attributes: DataFrame containing the attribute taxonomy.
    :return: DataFrame with recoded attributes.
    """
    # Recode the 'Education' column based on the values of '129403'
    df['Education'] = np.where(df['129403'].isin([131937, 131938, 131939]), 'low',
                               np.where(df['129403'].isin([131936, 131940]), 'high', df['129403']))
    # Create a new column for each unique value in 'Education'
    for val in df['Education'].unique():
        col_name = f"Education_{val}"
    df[col_name] = np.where(df['Education'] == val, 1, 0)

    df.drop(columns=['129403', 'Education'], inplace=True)

    # Recode the 'child18' column based on the values of '288472'
    df['child18'] = np.where(df['288472'] == 288473, 1,
                             np.where(df['288472'] == 288474, 0, df['288472']))

    df.drop(columns=['288472'], inplace=True)


    df['HHincome'] = np.where(df['422692'] == 422693, "<$25k",
                              np.where(df['422692'] == 422694, "$25k-50k",
                                       np.where(df['422692'] == 422695, "$50k-75k",
                                                np.where(df['422692'] == 422696, "$75k-100k",
                                                         np.where(df['422692'] == 422697, "$100k-150k",
                                                                  np.where(df['422692'] == 422698, "$150k-200k",
                                                                           np.where(df['422692'] == 422699, "$75k-100k",
                                                                                    np.where(df['422692'] == 422700, "$250k+", df['422692']))))))))

    # Create a new column for each unique value in 'HHincome'
    for val in df['HHincome'].unique():
        col_name = f"HHincome_{val}"
    df[col_name] = np.where(df['HHincome'] == val, 1, 0)
    # Drop '422692' and 'HHincome' columns
    df.drop(columns=['422692', 'HHincome'], inplace=True)


    df['AgeGroup'] = np.where(df['422752'] == 422753, "18-24",
                              np.where(df['422752'] == 422755, "25-34",
                                       np.where(df['422752'] == 422754, "35-44",
                                                np.where(df['422752'] == 422005, "45-49",
                                                         np.where(df['422752'] == 422006, "50-54",
                                                                  np.where(df['422752'] == 422756, "55-64",
                                                                           np.where(df['422752'] == 422757, "65+", df['422752'])))))))

    # Create a new column for each unique value in 'AgeGroup'
    for val in df['AgeGroup'].unique():
        col_name = f"AgeGroup_{val}"
    df[col_name] = np.where(df['AgeGroup'] == val, 1, 0)

    # Drop '422752' and 'AgeGroup' columns
    df.drop(columns=['422752', 'AgeGroup'], inplace=True)

    # recode alcohol attributes
    df['orderedAlcoholOnline'] = np.where(df['319501'] == 319503, 1,
                                          np.where(df['319501'].isin([319502, 322999]), 0, df['319501']))
    df.drop(columns=['319501'], inplace=True)
    df['consumeOffPremise'] = np.where(df['368639'] == 368640, 1,
                                       np.where(df['368639'].isin([368641, 370424]), 0, df['368639']))
    df.drop(columns=['368639'], inplace=True)
    df['wine3mosIncrease'] = np.where(df['368669'] == 368672, 1,
                                      np.where(df['368669'].isin([368670, 368671, 368673]), 0, df['368669']))
    df.drop(columns=['368669'], inplace=True)
    df['liquor3mosIncrease'] = np.where(df['368674'] == 368677, 1,
                                        np.where(df['368674'].isin([368675, 368676, 368678]), 0, df['368674']))
    df.drop(columns=['368674'], inplace=True)
    df['beer3mosIncrease'] = np.where(df['397595'] == 397598, 1,
                                      np.where(df['397595'].isin([397596, 397597, 397599]), 0, df['397595']))
    df.drop(columns=['397595'], inplace=True)
    df['likelyAlcoholOnline'] = np.where(df['387406'] == 387407, 1,
                                         np.where(df['387406'].isin([387408, 387409, 387410]), 0, df['387406']))
    df.drop(columns=['387406'], inplace=True)
    df['decisioner'] = np.where(df['395143'] == 395144, 1,
                                np.where(df['395143'].isin([395145, 395146]), 0, df['395143']))
    df.drop(columns=['395143'], inplace=True)

    df['liquorQuality'] = np.where(df['397674'] == 397675, "luxury",
                                   np.where(df['397674'] == 397676, "topShelf",
                                            np.where(df['397674'] == 397677, "midShelf",
                                                     np.where(df['397674'] == 397678, "value",
                                                              np.where(df['397674'] == 397679, "notApplicable", df['397674'])))))

    for val in df['liquorQuality'].unique():
        col_name = f"liquorQuality_{val}"
    df[col_name] = np.where(df['liquorQuality'] == val, 1, 0)

    df.drop(columns=['397674', 'liquorQuality'], inplace=True)

    df['willingBurbon'] = np.where(df['397680'].isin([397684,397686]), 1,
                                   np.where(df['397680'].isin([397681, 397682, 397683]), 0, df['397680']))
    df.drop(columns=['397680'], inplace=True)

    df['wineHolidayPurch'] = np.where(df['399194'] == 399195, 1,
                                      np.where(df['399194'] == 399229, 0, df['399194']))
    df.drop(columns=['399194'], inplace=True)

    df.drop(columns=['418846'], inplace=True)
    ### drop '418846' column (duplicate of '395143')

    df['past3mosWine'] = np.where(df['426033'] == 426034, "daily",
                                  np.where(df['426033'] == 426035, "manyWeekly",
                                           np.where(df['426033'] == 426036, "onceWeekly",
                                                    np.where(df['426033'] == 426037, "onceMonthly",
                                                             np.where(df['426033'] == 426038, "lessThan1monthly",
                                                                      np.where(df['426033'] == 426039, "none",
                                                                               np.where(df['426033'] == 426040, "DntKnw",
                                                                                        np.where(df['426033'] == 426041, "notApplicable", df['426033']))))))))

    # Create a new column for each unique value in 'past3mosWine'
    for val in df['past3mosWine'].unique():
        col_name = f"past3mosWine_{val}"
    df[col_name] = np.where(df['past3mosWine'] == val, 1, 0)
    # Drop '426033' and 'past3mosWine' columns
    df.drop(columns=['426033', 'past3mosWine'], inplace=True)

    return df
def save_data_to_csv(data, output_dir, filename):
    filepath = os.path.join(output_dir, filename)
    data.to_csv(filepath, index=False)

##############################################################################
if __name__ == '__main__':  # Run the main function
    main()
##############################################################################
##############################################################################


