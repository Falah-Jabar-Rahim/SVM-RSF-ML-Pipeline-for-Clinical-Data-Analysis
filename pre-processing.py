import os
import argparse
import numpy as np
from util import Feature_Normalization, const_fill_missing, read_data, feature_plot, data_imputation


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='data/RSF_data.xlsx', help='path file containing data')
parser.add_argument('--output_dir', type=str, default='output', help='path to output directory')
parser.add_argument('--features_to_normalize', type=str, nargs='*',
                    default=['Sex', 'CTPs', 'Nodule_No', 'Multinodular_tumor', 'BCLC', 'Age', 'BMI', 'MTD', 'ALT',
                             'AST', 'ALP', 'ALB', 'Bili', 'WCC', 'Hb', 'Neu', 'Lym', 'PLT', 'CRP', 'AFP', 'ALBIs',
                             'ALBIg'], help='features to normalize e.g., [f1, f2, f5] or []))')
parser.add_argument('--scaling_method', type=str, default='MinMaxScaler',
                    help='scaling_method: MinMaxScaler or StandardScaler')
parser.add_argument('--imputation_method', type=str, default="MICE", help='imputation method "const" or "MICE" ')
parser.add_argument('--replace_val', type=int, default=-1, help='replace missing values (e.g., -1 or any value)')
parser.add_argument('--add_indicator', type=bool, default=False, help='add indicator colum or not')



if __name__ == "__main__":
    np.random.seed(42)
    # Parse arguments
    args = parser.parse_args()

    # Data-related settings
    data_path = args.input_dir
    output_path = args.output_dir
    os.makedirs(output_path, exist_ok=True)
    features_to_normalize = args.features_to_normalize
    scaling_method = args.scaling_method
    imputation_method = args.imputation_method
    replace_val = args.replace_val
    add_indicator = args.add_indicator

    # read data
    feature = read_data(data_path)

    ## ----- 1: plot features -----
    feature_plot(feature.copy(), output_path)

    ## ----- 2: fill missing values  -----
    if imputation_method == "const":
        feature_filled = const_fill_missing(feature, replace_val, add_indicator)
    else:
        feature_filled = data_imputation(feature)

    ## ----- 3: feature normalization  -----
    feature_filled_norm = Feature_Normalization(feature_filled, features_to_normalize, scaling_method)

    ## ----- 4: save processed data  -----
    feature_filled_norm.to_excel(f"{output_path}/feature_filled_norm.xlsx", index=False)
    print(f"processed files saved in {output_path}/feature_filled_norm.xlsx")
