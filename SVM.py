import argparse
import numpy as np
from util import classification_plots, read_data, SVM_model

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='output/test2.xlsx', help='path to processed file containing data')
parser.add_argument('--output_dir', type=str, default='output', help='path to output directory')
parser.add_argument('--kernel_fn', type=str, default='rbf', help='option:linear, rbf, poly, sigmoid')
parser.add_argument('--C', type=float, default=1, help='regularization parameter')
parser.add_argument('--gamma', type=float, default=0.55, help='Option: scale, auto')

if __name__ == "__main__":

    np.random.seed(42)
    # Parse arguments
    args = parser.parse_args()
    data_path = args.dataset

    ## --------- model traning and testing ----------------------------
    print("Model training and testing with 5K-fold cross validation")
    processed_feature = read_data(data_path)
    results, importance_score = SVM_model(args, processed_feature)
    classification_plots(args, results, importance_score)
    print(f"Results are saved in {args.output_dir} folder")


