

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
    args = parse_args()
    fs = s3fs.S3FileSystem(anon=False)
    # Read the data from S3
    tobacco_data = read_data_from_s3(fs, args.tobacco_file_path, 251)
    tobacco_data = tobacco_data.drop_duplicates(subset='RID')
    newX = split_bottleneck(tobacco_data)
    pca = get_pca(newX, 12)
    pca_data = apply_pca_with_nulls(newX, pca)
    newX = pca_data.copy()
    newX.to_csv(args.s3_output_directory + 'tobacco_pca_data.csv', index=False)
    # Recode Resonate attributes with the help of the attribute taxonomy
    # Load the attribute taxonomy
    # attribute_taxonomy = pd.read_csv(args.local_directory + 'attribute_taxonomy.csv')
    attributes = attributes_to_recode(newX)
    recoded_data = recode_attributes(newX, attributes)
    save_data_to_csv(recoded_data, args.s3_output_directory, 'tobacco_blca_input.csv')
    save_data_to_csv(recoded_data, args.local_directory, 'tobacco_blca_input.csv')
    inputFilePath = args.local_directory + 'tobacco_blca_input.csv'
    outputPath = args.local_directory
    colsToDrop = 'RID'
    arguments = [inputFilePath, outputPath, args.numSegments, args.burnIn, args.thin, args.iterations, colsToDrop, args.useCase]
    blcaRcodePath = '/Users/samhawala/Documents/GitHub/guardians-segmentation/blcaExperiments/tobaccoExperiment/src/R/blca.R'
    os.system('Rscript ' + blcaRcodePath + ' ' + ' '.join(arguments))
    # sync to s3
    print('Syncing to s3...')
    os.system('aws s3 sync ' + output_dir + ' ' + args.s3_output_directory)
########################################################################################################

######################################################################################################################
########################################################################################################
parser = argparse.ArgumentParser(description='Experiment TOBACCO Segmentation')

parser.add_argument('--useCase', type=str,
                    default="tobacco",
                    help='The current use-case for segmentation')

parser.add_argument('--tobacco_file_path', type=str,
                    default='s3://datasci-scantle/disruptCRM/bsm/tobacco/',
                    help='The location of the AUTO files')

parser.add_argument('--numSegments', type=str,
                    default='5',
                    help='The default number of desired segments')

parser.add_argument('--burnIn', type=str,
                    default='100',
                    help='The number of initial iterations to discard as burn-in')

parser.add_argument('--thin', type=str,
                    default='0.5',
                    help='The thinning rate to achieve good mixing')

parser.add_argument('--iterations', type=str,
                    default='1000',
                    help='The number of iterations to run the gibbs sampler for after burn-in')

parser.add_argument('--tobacco_file_path', type=str,
                    default='s3://datasci-scantle/disruptCRM/bsm/tobacco/',
                    help='The location of the AUTO files')

parser.add_argument('--local_directory', type=str,
                    default='/Users/samhawala/Documents/work2024/Segmentations/finalExperiments/tobaccoExperiment/',
                    help='The location of the attribute hierarchy file')

parser.add_argument('--s3_output_directory', type=str,
                    default='s3://resonate-datasci-dev/shawala/Segmentations/tobaccoExperiment/',
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
    # Recode Education column
    newX['Education'] = np.where(newX['X129403'].isin([131937, 131938, 131939]), 'low',
                                 np.where(newX['X129403'].isin([131936, 131940]), 'high', newX['X129403']))

    newX = pd.get_dummies(newX, columns=['Education'], prefix='Education')
    newX.drop(columns=['X129403'], inplace=True)

    # Recode child18 column
    newX['child18'] = np.where(newX['X288472'] == 288473, 1,
                               np.where(newX['X288472'] == 288474, 0, newX['X288472']))
    newX.drop(columns=['X288472'], inplace=True)

    # Recode and one-hot encode specific columns
    for col in ['X391882', 'X391894', 'X395139', 'X420664', 'X420741', 'X422752']:
        newX = pd.get_dummies(newX, columns=[col], prefix=col)

    # Recode vehicle_Year column
    newX['vehicle_Year'] = np.where(newX['X425579'].isin([425580, 425581, 425582, 425583, 425584, 425585, 425586]), '2013orOlder',
                                    np.where(newX['X425579'].isin([425587, 425588, 425589, 425590, 425591]), '2014_2019',
                                             np.where(newX['X425579'].isin([425592, 425593, 425594, 425595, 425596]), '2020+',
                                                      np.where(newX['X425579'].isin([425597, 425598]), 'dontKnowNA', newX['X425579']))))

    newX = pd.get_dummies(newX, columns=['vehicle_Year'], prefix='vehicle_Year')
    newX.drop(columns=['X425579'], inplace=True)

    # Recode currentCar column
    newX['currentCar'] = np.where(newX['X130469'] == 145511, 'Used',
                                  np.where(newX['X130469'] == 145512, 'New',
                                           np.where(newX['X130469'] == 145513, 'dontKnow',
                                                    np.where(newX['X130469'] == 145514, 'Lease',
                                                             np.where(newX['X130469'] == 383865, 'notAplcbl', newX['X130469'])))))

    newX = pd.get_dummies(newX, columns=['currentCar'], prefix='currentCar')
    newX.drop(columns=['X130469'], inplace=True)

    # Recode carCost column
    newX['carCost'] = np.where(newX['X391834'].isin([391838, 391839]), '45k+',
                               np.where(newX['X391834'].isin([391835, 391836, 391837, 391840, 391841]), 'under45k', newX['X391834']))
    newX = pd.get_dummies(newX, columns=['carCost'], prefix='carCost')
    newX.drop(columns=['X391834'], inplace=True)

    # Similar logic for the other columns
    newX['newCarCost'] = np.where(newX['X155046'].isin([160628, 160629]), '45k+',
                                  np.where(newX['X155046'].isin([160625, 160626, 160627, 160630]), 'under45k', newX['X155046']))
    newX = pd.get_dummies(newX, columns=['newCarCost'], prefix='newCarCost')
    newX.drop(columns=['X155046'], inplace=True)

    # Continue similarly with other columns in a loop
    for col, recodes in {
        'whenNextCar': {160808: 'within3mos', 160809: '4-6mos', 160810: '6-12mos', 160811: '1-2yrs', 160812: '2yrs+'},
        'carNum': {196929: 'none', 196930: 'one', 196931: 'two'},
        'hybrid': {207177: 'no', 207605: 'no', 207632: 'no', 268987: 'no', 207481: 'yes', 207614: 'yes'},
        'plg_hybrid': {286919: 'no', 207324: 'no', 207789: 'no', 207921: 'no', 206941: 'yes', 207932: 'yes'},
        'electric': {206967: 'no', 207229: 'no', 207895: 'no', 268988: 'no', 206966: 'yes', 207915: 'yes'},
        'carElectric': {207042: 'no', 207111: 'no', 208123: 'no', 207598: 'partialOrfull', 207636: 'partialOrfull', 383864: 'partialOrfull'},
        'newCar6mos': {391914: 'no', 391915: 'no', 391913: 'yes'},
        'contactless': {391929: 'notOpenTo', 391930: 'notOpenTo', 391931: 'notOpenTo', 391932: 'openTo', 391933: 'openTo'}
    }.items():
        newX[col] = newX[col].map(recodes)
        newX = pd.get_dummies(newX, columns=[col], prefix=col)

    # Final binary recodes
    newX['intend'] = np.where(newX['X394288'] == 394293, 1, np.where(newX['X394288'] == 394323, 0, newX['X394288']))
    newX['decisioner'] = np.where(newX['X418906'] == 418907, 1, np.where(newX['X418906'] == 418908, 0, newX['X418906']))
    newX.drop(columns=['X394288', 'X418906'], inplace=True)

    # Drop unnecessary column
    newX.drop(columns=['X420675'], inplace=True)

    # Recode AgeGroup column
    newX['AgeGroup'] = newX['X422752'].map({422753: '18-24', 422755: '25-34', 422754: '35-44', 422005: '45-49', 422006: '50-54', 422756: '55-64', 422757: '65+'})
    newX = pd.get_dummies(newX, columns=['AgeGroup'], prefix='AgeGroup')
    newX.drop(columns=['X422752'], inplace=True)

    # Recode HHincome column
    newX['HHincome'] = newX['X422692'].map({422693: '<$25k', 422694: '$25k-50k', 422695: '$50k-75k', 422696: '$75k-100k',
                                            422697: '$100k-150k', 422698: '$150k-200k', 422699: '$75k-100k', 422700: '$250k+'})
    newX = pd.get_dummies(newX, columns=['HHincome'], prefix='HHincome')
    newX.drop(columns=['X422692'], inplace=True)

    # Final binary recodes for motorcycles
    newX['motorcycles'] = np.where(newX['X423714'] == 423715, 1, np.where(newX['X423714'] == 423716, 0, newX['X423714']))
    newX.drop(columns=['X423714'], inplace=True)

def save_data_to_csv(data, output_dir, filename):
    filepath = os.path.join(output_dir, filename)
    data.to_csv(filepath, index=False)

##############################################################################
if __name__ == '__main__':  # Run the main function
    main()
##############################################################################
##############################################################################
