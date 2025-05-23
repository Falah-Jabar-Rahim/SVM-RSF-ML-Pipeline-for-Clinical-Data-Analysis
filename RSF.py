import argparse
import numpy as np
from util import read_data, RSF_model, classification_plots, RFS_plots

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='output/test.xlsx', help='path to processed file containing data')
parser.add_argument('--output_dir', type=str, default='output', help='path to output directory')
parser.add_argument('--lts', type=float,  default=15, help='number of death up to which months')
parser.add_argument('--t_calb', type=float,  default=36, help='month at which to compute and plot calibration')
parser.add_argument('--t_youden', type=float,  default=24, help='month at which to compute Youden threshold')

# model parameters
parser.add_argument('--n_estimators', type=int, default=300, help='Number of trees in the random survival forest')
parser.add_argument('--min_samples_split', type=int, default=10, help='Minimum number of samples required to split an internal node')
parser.add_argument('--min_samples_leaf', type=int, default=10, help='Minimum number of samples required to be at a leaf node')
parser.add_argument('--max_features', type=str, default='sqrt', help='Number of features to consider at each split (e.g., "sqrt", "log2", or float)')
parser.add_argument('--n_splits', type=int, default=5, help='Number of folds for cross-validation')


if __name__ == "__main__":

    np.random.seed(42)
    # Parse arguments
    args = parser.parse_args()
    # Data-related settings
    data_path = args.dataset
    feature = read_data(data_path)
    print("Model training and testing with 5K-fold cross validation")
    plt_results, x, y, ID_time_event = RSF_model(args, feature)

    RFS_plots(args, plt_results, x, y, ID_time_event)
    print(f"Results are saved in {args.output_dir} folder")

