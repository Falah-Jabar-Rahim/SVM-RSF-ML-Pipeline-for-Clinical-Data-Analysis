import os
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sksurv.util import Surv
from scipy.stats import ttest_ind
from sklearn.model_selection import KFold
from lifelines import KaplanMeierFitter
from statsmodels.imputation.mice import MICEData
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from matplotlib.ticker import MultipleLocator
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sksurv.ensemble import RandomSurvivalForest
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sksurv.metrics import concordance_index_censored, brier_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve, average_precision_score, roc_auc_score


def RSF_model(args, feature, plt_results=None):
    n_estimators = args.n_estimators
    min_samples_split = args.min_samples_split
    min_samples_leaf = args.min_samples_leaf
    max_features = args.max_features
    n_splits = args.n_splits

    if plt_results is None:
        plt_results = build_plot_results_dict_rf()

    # identify column names by position
    id_col = feature.columns[0]  # ID
    time_col = feature.columns[-2]  # time
    event_col = feature.columns[-1]  # event
    # create survival target y
    feature_tmp = feature.copy()
    feature_tmp[event_col] = feature_tmp[event_col].astype(bool)
    y = Surv.from_dataframe(event_col, time_col, data=feature_tmp)
    # drop ID, time, and event columns to get features
    x = feature.drop(columns=[id_col, time_col, event_col])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # === Store results ===
    c_indices = []
    all_risk_scores = []
    brier_scores = []
    feature_impts = []
    surv_funcs_all = []
    surv_prob_all = []
    true_event_times = []
    all_best_threshold = []

    # === Run K-Fold CV ===
    for fold, (train_idx, test_idx) in enumerate(kf.split(x), 1):
        X_train, X_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        rsf = RandomSurvivalForest(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=-1,
            random_state=fold)

        rsf.fit(X_train, y_train)
        # === Risk Scores ===
        risk_scores = np.array(rsf.predict(X_test))
        all_risk_scores.extend(risk_scores)
        # === Concordance Index ===
        c_index = np.array(
            concordance_index_censored(y_test[event_col], y_test[time_col], risk_scores)[0])
        c_indices.append(c_index)
        # === Predict survival functions ===
        surv_funcs_list = rsf.predict_survival_function(X_test)
        surv_funcs_all.extend(surv_funcs_list)
        surv_probs = np.array(rsf.predict_survival_function(X_test, return_array=True))
        surv_prob_all.extend(surv_probs)
        # === Time Grid (grab once) ===
        if fold == 1:
            sample_fn = rsf.predict_survival_function(X_test[:1])[0]
            true_event_times = sample_fn.x

        # Feature importance in each fold:
        importance = permutation_importance(rsf, X_test, y_test, n_repeats=10, random_state=fold)
        # After computing permutation importance
        feature_names = X_test.columns.tolist()
        importance_scores = importance.importances_mean
        # Create a dictionary: {feature_name: score}
        feature_importance_dict = dict(zip(feature_names, importance_scores))
        # Store this dictionary (or append to a list of dicts)
        feature_impts.append(feature_importance_dict)

        # Compute the cutoff to separate the groups
        best_threshold = compute_youden_threshold(args, risk_scores, y_test[time_col], y_test[event_col])
        all_best_threshold.append(best_threshold)

        brier_scores_at_times, times, status = compute_brier_score_safely(rsf, X_test, y_train, y_test,
                                                                          [id_col, time_col, event_col],
                                                                          fold)
        if status != "skipped":
            approx_ibs = np.mean(brier_scores_at_times)
            brier_scores.append(approx_ibs)

    feature_impt_df = pd.DataFrame(feature_impts)  # rows = folds, columns = features
    avg_importance = feature_impt_df.mean().to_dict()
    plt_results["all_risk_scores"] = all_risk_scores
    plt_results["true_event_times"] = true_event_times
    plt_results["all_c_indices"] = np.mean(c_indices)
    plt_results["IB_scores"] = np.mean(brier_scores)
    plt_results["feature_impt"] = avg_importance
    plt_results["surv_funcs_all"] = surv_funcs_all
    plt_results["surv_prob_all"] = surv_prob_all
    plt_results["best_threshold"] = np.mean(all_best_threshold)

    return plt_results, x, y, [id_col, time_col, event_col]


def compute_youden_threshold(args, risk_scores, durations, events, do_plot=False):
    t_star = args.t_youden
    # Binary outcome at t_star
    binary_event = (durations <= t_star) & (events == 1)
    # Compute ROC
    fpr, tpr, thresholds = roc_curve(binary_event, risk_scores)

    # Youden's J
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_idx]

    # ---- plot ROC curve ------
    if do_plot:
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label="ROC Curve", color="blue")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Chance")
        # Highlight Youden's J point
        plt.scatter(fpr[best_idx], tpr[best_idx], color="red", label=f"Threshold = {best_threshold:.2f}")
        plt.text(fpr[best_idx], tpr[best_idx] - 0.05, f"J = {tpr[best_idx] - fpr[best_idx]:.2f}", fontsize=10)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve at t={24} months")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return best_threshold


def compute_brier_score_safely(rsf, X_test, y_train, y_test, ID_time_event, fold=0, n_times=100):
    """
    Safely computes the integrated Brier score using RSF survival predictions,
    with fallback for invalid time ranges or exceptions.

    Returns:
        - brier_scores_at_times: list of Brier scores at sampled time points
        - times: array of time points used
        - status: "ok", "fallback", or "skipped"
    """

    # Step 1: Determine StepFunction time domain
    surv_fns = rsf.predict_survival_function(X_test)
    fn_min = max(fn.domain[0] for fn in surv_fns)
    fn_max = min(fn.domain[1] for fn in surv_fns)

    # Step 2: Compute safe [t_min, t_max] range
    t_min = max(
        y_train[ID_time_event[1]].min(),
        y_test[ID_time_event[1]].min(),
        fn_min
    )
    t_max = min(
        y_train[ID_time_event[1]].max(),
        y_test[ID_time_event[1]].max(),
        fn_max
    ) - 1e-4

    # Step 3: Check if time window is valid
    if t_max <= t_min:
        print(f"Fold {fold}: Skipping IBS — No safe time window.")
        return None, None, "skipped"

    # Step 4: Build time grid
    times = np.linspace(t_min, t_max, n_times)

    # Step 5: Try full Brier score calculation
    try:
        surv_funcs_array = np.asarray([[fn(t) for t in times] for fn in surv_fns])
        _, brier_scores_at_times = brier_score(y_train, y_test, surv_funcs_array, times)
        return brier_scores_at_times, times, "ok"

    except ValueError as e:
        print(f"Fold {fold}: full Brier failed — trying partial fallback.")

        # Step 6: Fallback — compute pointwise
        brier_scores_at_times = []
        valid_times = []

        for t in times:
            try:
                surv_point = np.array([[fn(t)] for fn in surv_fns])
                _, bs = brier_score(y_train, y_test, surv_point, [t])
                brier_scores_at_times.append(bs[0])
                valid_times.append(t)
            except ValueError:
                continue

        if len(brier_scores_at_times) == 0:
            print(f"Fold {fold}: All Brier attempts failed — skipping.")
            return None, None, "skipped"

        return brier_scores_at_times, valid_times, "fallback"


def build_plot_results_dict_rf():
    base_structure = {
        "all_risk_scores": [],
        "true_event_times": [],
        "all_c_indices": [],
        "IB_scores": [],
        "feature_impt": [],
        "surv_funcs_all": [],
        "surv_prob_all": [],
        "best_threshold": []
    }
    return base_structure


def RFS_plots(args, plt_results, x, y, ID_time_event, p_val_pos=(60, 0.7), n_bins=5, pnt_num=200, figsize=(8, 6),
              fontsize=12, dpi=600):
    lts = args.lts
    t_star = args.t_calb

    output_path = args.output_dir
    np.random.seed(42)
    kmf = KaplanMeierFitter()
    durations = y[ID_time_event[1]]
    events = y[ID_time_event[2]]
    times = plt_results["true_event_times"]
    surv_fns = plt_results["surv_funcs_all"]
    risk_scores = np.array(plt_results["all_risk_scores"])
    # shared time grid
    fn_min = max(fn.domain[0] for fn in surv_fns)
    fn_max = min(fn.domain[1] for fn in surv_fns)
    shared_times = np.linspace(fn_min, fn_max, pnt_num)
    surv_prob_all = np.array([[fn(t) for t in shared_times] for fn in surv_fns])

    # Risk groups
    best_threshold = plt_results["best_threshold"]
    # High risk = score above or equal to threshold
    high_risk_mask = risk_scores >= best_threshold
    # Low risk = score below threshold
    low_risk_mask = risk_scores < best_threshold
    mean_surv_low = np.median(surv_prob_all[low_risk_mask], axis=0)
    mean_surv_high = np.median(surv_prob_all[high_risk_mask], axis=0)

    # === Survival Plot ===
    plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize=figsize, dpi=dpi)
    for i in range(len(surv_fns)):
        plt.step(shared_times, surv_prob_all[i], where="post", color="gray", alpha=0.7, linewidth=1)
        if events[i] == 1:
            time = durations[i]  # time of event
            prob_at_time = surv_fns[i](time)
            plt.plot(time, prob_at_time, 'ro', markersize=2)

    plt.plot([], [], color="gray", alpha=0.7, linewidth=1, label="Individual Patients")

    # compute overall median
    median_all = np.median(surv_prob_all, axis=0)
    plt.step(shared_times, median_all, label="Median", color="black", linestyle="-", linewidth=2.5)

    plt.step(shared_times, mean_surv_low, where="post", color="green", linewidth=2.5,
             label=f"Low Risk (median, n={np.sum(low_risk_mask)})")
    plt.step(shared_times, mean_surv_high, where="post", color="red", linewidth=2.5,
             label=f"High Risk (median, n={np.sum(high_risk_mask)})")

    plt.plot([], [], 'ro', markersize=2, label='Death')

    plt.xlabel("Months", fontweight='bold', fontsize=fontsize)
    plt.ylabel("Survival Probability", fontweight='bold', fontsize=fontsize)
    plt.grid(False)
    plt.xlim(0, fn_max)

    # Log-rank test between risk groups
    durations_low = durations[low_risk_mask]
    events_low = events[low_risk_mask]
    durations_high = durations[high_risk_mask]
    events_high = events[high_risk_mask]

    # For low risk group
    # 1. Dead up to time 24
    dead_up_to_24_low = np.sum((durations_low <= lts) & (events_low == 1))

    # 2. Dead after time 24
    dead_after_24_low = np.sum((durations_low > lts) & (events_low == 1))

    # For high risk group
    # 1. Dead up to time 24
    dead_up_to_24_high = np.sum((durations_high <= lts) & (events_high == 1))

    # 2. Dead after time 24
    dead_after_24_high = np.sum((durations_high > lts) & (events_high == 1))
    # statistical test using t-test
    t_stat, p_value = ttest_ind(mean_surv_low, mean_surv_high, equal_var=False)  # Welch’s correction

    plt.text(p_val_pos[0], p_val_pos[1], f" p-value = {p_value:.0e}", fontsize=fontsize)
    plt.legend()
    plt.xlim(0, fn_max)
    plt.tight_layout()
    survival_plot_path = os.path.join(output_path, "rsf_survival_probability.png")
    plt.savefig(survival_plot_path)
    plt.close()

    summary_rows = {
        "C-Index": plt_results["all_c_indices"],
        "IBS": plt_results["IB_scores"],
        f"dead_up_to_{lts}_low": dead_up_to_24_low,
        f"dead_after_{lts}_low": dead_after_24_low,
        f"dead_up_to_{lts}_high": dead_up_to_24_high,
        f"dead_after_{lts}_high": dead_after_24_high,
    }

    ## --------- plot of feature importance ---------
    impt = plt_results["feature_impt"]
    impt_series = pd.Series(impt).sort_values()

    # Plot
    plt.figure(figsize=figsize, dpi=dpi)
    plt.barh(impt_series.index, impt_series.values, color="steelblue", edgecolor="black", alpha=0.85)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel("Importance score", fontsize=fontsize, fontweight="bold")
    plt.ylabel("Features", fontsize=fontsize, fontweight="bold")

    # dynamic x-axis range
    min_val = impt_series.min()
    max_val = impt_series.max()
    range_padding = (max_val - min_val) * 0.1
    plt.xlim(min_val - range_padding, max_val + range_padding)

    # Tick interval
    tick_step = round((max_val - min_val) / 5, 3)
    tick_step = max(tick_step, 0.001)
    plt.gca().xaxis.set_major_locator(MultipleLocator(tick_step))

    # Save
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"rsf_feature_importance.png"))
    plt.close()

    drop_values_dead = []
    drop_values_cens = []
    for i in range(len(surv_fns)):

        t_event = min(max(durations[i], surv_fns[i].domain[0]), surv_fns[i].domain[1])
        surv_fn = surv_fns[i]
        s_at_event = surv_fn(t_event)
        times_after = shared_times[(shared_times > t_event)]

        if len(times_after) > 0:
            mean_after = np.mean([surv_fn(t) for t in times_after])
            drop = s_at_event - mean_after
        else:
            drop = np.nan

        if events[i] == 1:
            drop_values_dead.append(drop)
        else:
            drop_values_cens.append(drop)
    avg_drop_dead = np.nanmean(drop_values_dead)
    avg_drop_cens = np.nanmean(drop_values_cens)

    summary_rows["avg_drop_dead"] = avg_drop_dead
    summary_rows["avg_drop_censor"] = avg_drop_cens
    summary_row = [summary_rows]
    summary_df = pd.DataFrame(summary_row)
    summary_path = os.path.join(output_path, "rsf_metrics.xlsx")
    summary_df.to_excel(summary_path, index=False)

    # --------- calibration Plot ---------
    plt.figure(figsize=figsize, dpi=dpi)
    surv_fns = plt_results["surv_funcs_all"]
    predicted_surv = np.array([fn(t_star) for fn in surv_fns])
    bin_edges = np.quantile(predicted_surv, np.linspace(0, 1, n_bins + 1))
    bin_ids = np.digitize(predicted_surv, bin_edges[1:-1])

    mean_pred = []
    mean_obs = []

    for b in range(n_bins):
        idx = (bin_ids == b)
        if np.sum(idx) == 0:
            continue
        pred_bin = predicted_surv[idx]
        dur_bin = durations[idx]
        event_bin = events[idx]
        mean_pred.append(np.mean(pred_bin))
        kmf.fit(dur_bin, event_observed=event_bin)
        mean_obs.append(kmf.predict(t_star))

    plt.plot(mean_pred, mean_obs, marker='o', color="green", label=f": IBS {plt_results['IB_scores']:.2f}")

    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    plt.xlabel("Predicted Survival Probability", fontweight='bold', fontsize=fontsize)
    plt.ylabel("Observed Survival Probability", fontweight='bold', fontsize=fontsize)
    plt.grid(True)
    plt.legend().set_visible(False)
    plt.tight_layout()
    calib_plot_path = os.path.join(output_path, "rsf_calibration.png")
    plt.savefig(calib_plot_path)
    plt.close()

def compute_feature_importance_lofo(X, y, base_model, scoring="roc_auc", cv=5):
    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Baseline performance with all features
    baseline_score = np.mean(cross_val_score(base_model, X, y, cv=cv_split, scoring=scoring))

    importances = []
    feature_names = X.columns

    for feature in feature_names:
        X_reduced = X.drop(columns=feature)
        model_clone = SVC(**base_model.get_params())  # re-initialize to avoid data leakage
        score = np.mean(cross_val_score(model_clone, X_reduced, y, cv=cv_split, scoring=scoring))
        importance = baseline_score - score
        importances.append(importance)

    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


def SVM_model(args, data_train, n_splits=5):
    # obtain model parameters
    kernel_fn = args.kernel_fn
    c = args.C
    gamma = args.gamma
    # train and test data
    y_train = data_train[data_train.columns[-1]]
    data_train = data_train.drop(columns=[data_train.columns[0], data_train.columns[-1]])

    svm = SVC(kernel=kernel_fn, C=c, gamma=gamma, probability=True, class_weight='balanced')
    # get predicted probabilities for classes
    print("prediction...")
    y_proba = cross_val_predict(svm, data_train, y_train, cv=n_splits, method='predict_proba')[:, 1]
    Acc, AUC, plt_results = classification_metrics(y_train, y_proba, args.output_dir)

    # compute LOFO feature importance
    print("computing feature importance...")
    importance = compute_feature_importance_lofo(data_train, y_train, svm, scoring="roc_auc")

    return plt_results, importance


def classification_metrics(y, y_proba, output_path, plt_results=None, bin_sz=5, figsize=(8, 6), dpi=600):
    if plt_results is None:
        plt_results = build_plot_results_dict()

    y_preds = np.array((y_proba >= 0.5).astype(int))
    # compute metrics
    Metrics = {'Acc': Accuracy_Metric(y, y_preds), 'F1': F1_Score(y, y_preds),
               'Precision': Precision_Metric(y, y_preds), 'Recall': Recall_Metric(y, y_preds)}
    cfm = Confusion_Matrix(y, y_preds)

    plt.figure(figsize=figsize)
    sns.heatmap(cfm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'])
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save the figure
    cm_path = os.path.join(output_path, f"svm_confusion_matrix.png")
    plt.savefig(cm_path, dpi=dpi)
    plt.close()

    # Convert confusion matrix to DataFrame (if 2D)
    if cfm.ndim > 1:
        for i in range(cfm.shape[0]):
            for j in range(cfm.shape[1]):
                Metrics[f'Confusion Matrix [{i},{j}]'] = [cfm[i, j]]

    metrics_df = pd.DataFrame(Metrics)

    # Plot calibration curve
    bins = np.linspace(0., 1. + 1e-8, bin_sz + 1)
    bin_ids = np.digitize(y_proba, bins) - 1

    bin_true = np.zeros(bin_sz)
    bin_pred = np.zeros(bin_sz)
    bin_count = np.zeros(bin_sz)

    for i in range(bin_sz):
        bin_mask = bin_ids == i
        bin_count[i] = np.sum(bin_mask)
        if bin_count[i] > 0:
            bin_true[i] = np.mean(y[bin_mask])
            bin_pred[i] = np.mean(y_proba[bin_mask])

    # Plot roc curve
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    # Calculate AUC and confidence interval
    auc_avg, ci_lower, ci_upper = bootstrap_auc(y, y_proba)

    # Confusion Matrix
    cm = confusion_matrix(y, y_preds)

    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y, y_proba)
    avg_precision = average_precision_score(y, y_proba)
    # compute Expected Calibration Error
    ece = compute_ece(y, y_proba)
    # Calibration Plot
    plt_results["calibration_plot"]["bin_pred"] = bin_pred
    plt_results["calibration_plot"]["bin_true"] = bin_true
    plt_results["calibration_plot"]["bin_count"] = bin_count
    plt_results["calibration_plot"]["ece"] = ece

    # ROC Curve
    plt_results["roc_curve"]["fpr"] = fpr
    plt_results["roc_curve"]["tpr"] = tpr
    plt_results["roc_curve"]["thresholds"] = thresholds
    plt_results["roc_curve"]["roc_auc"] = roc_auc
    plt_results["roc_curve"]["auc_avg"] = auc_avg
    plt_results["roc_curve"]["ci_lower"] = ci_lower
    plt_results["roc_curve"]["ci_upper"] = ci_upper

    # Confusion Matrix
    plt_results["confusion_matrix"]["cm"] = cm

    # Precision-Recall
    plt_results["precision_recall"]["precision"] = precision
    plt_results["precision_recall"]["recall"] = recall
    plt_results["precision_recall"]["avg_precision"] = avg_precision

    output_file = os.path.join(output_path, "svm_metrics.xlsx")

    # Save to Excel
    with pd.ExcelWriter(output_file) as writer:
        metrics_df.to_excel(writer, index=False)

    return Accuracy_Metric(y, y_preds), roc_auc, plt_results


def classification_plots(args, plt_results, feature_importanc, figsize=(8, 6), fontsize=12, dpi=600):
    np.random.seed(42)
    output_path = args.output_dir
    # ----------------- Plotting calibration curves -----------------------------#
    plt.figure(figsize=figsize, dpi=dpi)
    plt.rcParams['font.family'] = 'Arial'

    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration', linewidth=2)

    # Loop over each variant and plot if available
    bin_pred = plt_results["calibration_plot"]["bin_pred"]
    bin_true = plt_results["calibration_plot"]["bin_true"]
    ece = plt_results["calibration_plot"]["ece"]

    plt.plot(bin_pred, bin_true, marker='o', label=f"SVM (ECE={ece:.2f})", color="green", linewidth=2)

    # Labels and legend
    plt.xlabel("Mean Predicted Probability", fontweight='bold', fontsize=fontsize)
    plt.ylabel("Fraction of Positives", fontweight='bold', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    calib_plot_path = os.path.join(output_path, "svm_calibration")
    plt.savefig(calib_plot_path)
    plt.close()

    # ----------------- Plotting ROC curve -----------------------------#
    plt.figure(figsize=figsize, dpi=dpi)
    plt.rcParams['font.family'] = 'Arial'
    # Plot ROC curves
    roc_data = plt_results["roc_curve"]
    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]
    roc_auc = roc_data["roc_auc"]
    ci_lower = roc_data["ci_lower"]
    ci_upper = roc_data["ci_upper"]

    plt.plot(
        fpr, tpr,
        label=f"SVM (AUC={roc_auc:.2f}, CI [{ci_lower:.2f}, {ci_upper:.2f}])",
        color="green",
        linewidth=2
    )

    # Plot the diagonal (random classifier)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')

    # Style and labels
    plt.xlabel("False Positive Rate", fontweight='bold', fontsize=fontsize)
    plt.ylabel("True Positive Rate", fontweight='bold', fontsize=fontsize)
    plt.legend(loc='lower right', fontsize=fontsize)
    plt.grid(True)
    plt.tight_layout()

    # Save the ROC plot
    roc_plot_path = os.path.join(output_path, "svm_roc.png")

    plt.savefig(roc_plot_path)
    plt.close()

    # ----------------- Plotting confusion_matrix  -----------------------------#
    plt.figure(figsize=figsize, dpi=dpi)
    plt.rcParams['font.family'] = 'Arial'
    # Loop to generate confusion matrix plots
    cm = plt_results["confusion_matrix"]["cm"]

    cm_array = np.array(cm)
    sns.heatmap(cm_array, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=["0", "1"],
                yticklabels=["0", "1"])
    plt.title(f"Confusion Matrix - SVM", fontsize=fontsize, fontweight='bold')
    plt.xlabel("Predicted class", fontsize=fontsize)
    plt.ylabel("Actual class", fontsize=fontsize)
    plt.tight_layout()

    # Save each confusion matrix
    save_path = f"{output_path}svm_confusion_matrix.png"
    plt.savefig(save_path)
    plt.close()

    # Define your variants

    # ----------------- Plotting precision_recall curves -----------------------------#
    plt.figure(figsize=figsize, dpi=dpi)
    plt.rcParams['font.family'] = 'Arial'

    # Plot
    precision = plt_results["precision_recall"]["precision"]
    recall = plt_results["precision_recall"]["recall"]
    avg_precision = plt_results["precision_recall"]["avg_precision"]

    plt.plot(recall, precision, label=f"SVM (AP = {avg_precision:.2f})", color="green", linewidth=2)

    # Style the plot
    plt.xlabel("Recall", fontsize=fontsize, fontweight='bold')
    plt.ylabel("Precision", fontsize=fontsize, fontweight='bold')
    plt.title("Precision-Recall Curve", fontsize=fontsize, fontweight='bold')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(fontsize=fontsize)
    plt.grid(True)
    plt.tight_layout()

    # Save
    pr_curve_path = os.path.join(output_path, 'svm_precision_recall.png')

    plt.savefig(pr_curve_path)
    plt.close()

    # ----------------- Plot feature importance -----------------------------#

    plt.figure(figsize=figsize, dpi=dpi)
    plt.rcParams['font.family'] = 'Arial'

    feature_importanc = pd.Series(feature_importanc).sort_values()

    # Plot
    plt.barh(feature_importanc.index, feature_importanc.values, color="steelblue", edgecolor="black", alpha=0.85)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel("Importance scores", fontsize=fontsize, fontweight='bold')
    plt.ylabel("Features", fontsize=fontsize, fontweight='bold')
    pr_curve_path = os.path.join(output_path, 'svm_feature_importance.png')
    plt.tight_layout()
    # Set x-axis range and tick interval
    #plt.xlim(-0.0025, 0.035)
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.005))
    plt.savefig(pr_curve_path)
    plt.close()


def data_imputation(df, epc=20):
    # Exclude the first column (e.g., ID)
    df_impute = df.iloc[:, 1:].copy()

    # Initialize MICE
    mice_data = MICEData(df_impute)

    # Run MICE updates
    for _ in range(epc):
        mice_data.update_all()

    # Get the imputed data
    df_imputed = mice_data.data.copy()

    # Optionally reattach the ID column
    df_imputed.insert(0, df.columns[0], df.iloc[:, 0].values)

    return df_imputed


def bootstrap_auc(y_true, y_pred, n_bootstraps=1000, ci=0.95):
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    for i in range(n_bootstraps):
        # Bootstrap by sampling with replacement
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # Compute confidence interval
    confidence_lower = sorted_scores[int((1.0 - ci) / 2 * len(sorted_scores))]
    confidence_upper = sorted_scores[int((1.0 + ci) / 2 * len(sorted_scores))]
    return np.mean(bootstrapped_scores), confidence_lower, confidence_upper


def ci_bounds(scores, alpha=None):
    lower = np.percentile(scores, (1 - alpha) / 2 * 100)
    upper = np.percentile(scores, (1 + alpha) / 2 * 100)
    return np.mean(scores), lower, upper


def feature_plot(f, output_path, fontsize=14, dpi=600, thr=10):
    plt.rcParams['font.family'] = 'Times New Roman'
    renamed = {}
    missing_info = {}

    for col in f.select_dtypes(include=[np.number]).columns:
        # Normalize if max > 10 (Just to visualize)
        max_val = f[col].max()
        if max_val > thr:
            f[col] = thr * (f[col] - f[col].min()) / (max_val - f[col].min())
            renamed[col] = col + " *"

        # Compute missing % for each feature
        missing_pct = f[col].isna().mean() * 100
        missing_info[col] = f"{missing_pct:.1f}%" if missing_pct > 0 else ""

    f.rename(columns=renamed, inplace=True)

    # Update column names with missing percentage
    updated_cols = []
    for col in f.columns:
        orig_col = col.replace(" *", "")  # strip asterisk to match missing_info keys
        missing = missing_info.get(orig_col, '')
        label = f"{col}\n({missing})" if missing else col
        updated_cols.append(label)

    # Plot
    plt.figure(figsize=(max(8, len(f.columns) * 0.6), 6), dpi=dpi)
    sns.violinplot(data=f)
    plt.xticks(ticks=np.arange(len(f.columns)), labels=updated_cols, rotation=90, fontsize=fontsize * 0.9)
    plt.ylabel("Feature Values", fontsize=fontsize)
    plt.xlabel("Features", fontsize=fontsize)
    plt.title("Feature Distribution (* = scaled to [0,10], % missing in brackets)", fontsize=fontsize)
    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(output_path, 'Feature_plot.png')
    plt.savefig(save_path)
    plt.close()


def Confusion_Matrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    return np.array(conf_matrix)


def Accuracy_Metric(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    return np.array(accuracy)


def Precision_Metric(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    return np.array(precision)


def Recall_Metric(y_true, y_pred):
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    return np.array(recall)


def F1_Score(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='macro')
    return np.array(f1)


def Rmse(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())  # for one step prediction or one output neuron only


def compute_ece(y_true, y_prob, n_bins=10):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges, right=True)

    ece = 0.0
    for i in range(1, n_bins + 1):
        bin_mask = bin_indices == i
        bin_size = np.sum(bin_mask)

        if bin_size > 0:
            bin_confidence = np.mean(y_prob[bin_mask])
            bin_accuracy = np.mean(y_true[bin_mask])
            bin_error = abs(bin_confidence - bin_accuracy)
            ece += (bin_size / len(y_true)) * bin_error

    return ece


def read_data(file_path):
    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.csv':
        df = pd.read_csv(file_path)
    elif ext.lower() in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xlsx")

    return df


def const_fill_missing(df, val, add_indicator):
    pat_id = [df.columns[0]]  # First column is assumed to be ID

    for col in df.columns.tolist():  # Use a static list to avoid column mutation issues
        if col in pat_id:
            continue  # Skip ID column

        if df[col].isnull().any():
            if add_indicator:
                # Compute the indicator before filling
                missing_indicator = df[col].isnull().astype(int)

            # Fill missing values
            df[col] = df[col].fillna(val)

            if add_indicator:
                # Insert the indicator column right after the current column
                col_index = df.columns.get_loc(col)
                df.insert(col_index + 1, f"{col}_missing", missing_indicator)

    return df


def Feature_Normalization(df, features_to_normalize, scaling_method):
    features_to_normalize = [col for col in features_to_normalize if col in df.columns]
    #  Initialize the chosen scaler
    if scaling_method == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaling_method == 'StandardScaler':
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid scaling method. Use 'MinMaxScaler' or 'StandardScaler'.")
    # Step Apply the selected scaler to the specified features (if available)
    if features_to_normalize:
        df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])
    else:
        print("No valid features to normalize.")

    normalized_df = df.copy()
    return normalized_df


def build_plot_results_dict():
    # Base nested structure
    plt_results = {
        "calibration_plot": {
            "bin_pred": [],
            "bin_true": [],
            "bin_count": [],
            "ece": [],
        },
        "roc_curve": {
            "fpr": [],
            "tpr": [],
            "thresholds": [],
            "roc_auc": 0,
            "auc_avg": 0,
            "ci_lower": 0,
            "ci_upper": 0
        },
        "confusion_matrix": {
            "cm": [[0, 0], [0, 0]]
        },
        "precision_recall": {
            "precision": [],
            "recall": [],
            "avg_precision": 0
        }
    }
    return plt_results
