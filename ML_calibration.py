##############################################################Libraries##########################################################################
import sys
import os
import re
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import numpy as np
import pyxdf
import mne
from misc.xdf2mne import stream2raw
from misc.XDF_utils import get_parameters_from_xdf_stream
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['toolbar'] = 'None'
from typing import cast
mne.set_log_level(verbose="WARNING")
from misc.log import getLogger
logger = getLogger(__name__)
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.tangentspace import tangent_space
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mne.decoding import CSP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import mode
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('../misc'))
from pathlib import Path
#####################################################################Functions##########################################################################
def sliding_window_epochs(epochs, window, step):
    data = epochs.get_data()   # shape: (n_epochs, n_channels, n_times)
    X = []    
    for ep in data:
        n_times = ep.shape[1]
        for start in range(0, n_times - window + 1, step):
            X.append(ep[:, start:start+window])
    X = np.array(X)
    info = epochs.info
    new_epochs = mne.EpochsArray(X, info, tmin=0 )
    return new_epochs

def run_repeated_split(X_trials, y_trials, info, window, step, n_repeats=100, test_size=0.2, random_state=42, verbose=False, return_first_split_features=False):
    sss = StratifiedShuffleSplit(n_splits=n_repeats, test_size=test_size, random_state=random_state)
    acc_trial_all = []
    kappa_trial_all = []
    acc_window_all = []
    kappa_window_all = []
    cm_all = []
    first_split_feat_train = None
    first_split_y_train = None
    first_split_feat_test = None
    first_split_y_test = None

    for split, (train_idx, test_idx) in enumerate(sss.split(X_trials, y_trials), start=1):
        X_train_trials = X_trials[train_idx]
        y_train_trials = y_trials[train_idx]
        X_test_trials = X_trials[test_idx]
        y_test_trials = y_trials[test_idx]
        train_epochs = mne.EpochsArray(X_train_trials, info=info, verbose=False)
        test_epochs = mne.EpochsArray(X_test_trials, info=info, verbose=False)
        train_windows, train_trial_ids = sliding_window_epochs2(train_epochs, window, step)
        test_windows, test_trial_ids = sliding_window_epochs2(test_epochs, window, step)
        y_train = y_train_trials[train_trial_ids]
        y_test = y_test_trials[test_trial_ids]
        feat_train = riemann(train_windows.get_data(), 0)
        feat_test = riemann(test_windows.get_data(), feat_train[1])
        # LDA with shrinkage + scaling, wrapped in a pipeline
        clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto') )
        clf.fit(feat_train[0], y_train)
        pred = clf.predict(feat_test[0])
        if return_first_split_features and split == 1:
            first_split_feat_train = feat_train[0]
            first_split_y_train = y_train
            first_split_feat_test = feat_test[0]
            first_split_y_test = y_test
        # window-level metrics
        acc_window = accuracy_score(y_test, pred)
        kappa_window = cohen_kappa_score(y_test, pred)
        acc_window_all.append(acc_window)
        kappa_window_all.append(kappa_window)
        # trial-level metrics (majority vote)
        trial_predictions = []
        trial_labels = []
        for trial in np.unique(test_trial_ids):
            idx = test_trial_ids == trial
            trial_pred = mode(pred[idx], keepdims=False).mode
            trial_true = y_test_trials[trial]
            trial_predictions.append(trial_pred)
            trial_labels.append(trial_true)
        trial_predictions = np.array(trial_predictions)
        trial_labels = np.array(trial_labels)
        acc_trial = accuracy_score(trial_labels, trial_predictions)
        kappa_trial = cohen_kappa_score(trial_labels, trial_predictions)
        acc_trial_all.append(acc_trial)
        kappa_trial_all.append(kappa_trial)
        cm_all.append(confusion_matrix(trial_labels, trial_predictions, labels=[0, 1]))
    result = {"acc_trial": np.array(acc_trial_all), "kappa_trial": np.array(kappa_trial_all), "acc_window": np.array(acc_window_all), "kappa_window": np.array(kappa_window_all), "confusion_matrix": np.sum(cm_all, axis=0),}
    if return_first_split_features:
        result["first_split_feat_train"] = first_split_feat_train
        result["first_split_y_train"] = first_split_y_train
        result["first_split_feat_test"] = first_split_feat_test
        result["first_split_y_test"] = first_split_y_test
    return result

def sliding_window_epochs2(epochs, window, step):
    data = epochs.get_data()
    X = []
    trial_ids = []
    for trial_idx, ep in enumerate(data):
        n_times = ep.shape[1]
        for start in range(0, n_times - window + 1, step):
            X.append(ep[:, start:start+window] )
            trial_ids.append(trial_idx)
    X = np.array(X)
    info = epochs.info
    new_epochs = mne.EpochsArray(X, info, tmin=0)
    return new_epochs, np.array(trial_ids)

def compute_covariances(X, shrinkage=0.1):
    n_trials, n_channels, n_samples = X.shape
    Ctrain = np.zeros((n_trials, n_channels, n_channels))
    for i in range(n_trials):
        X_trial = X[i]
        C = np.cov(X_trial)
        # shrinkage toward scaled identity, proportional to the trial's own trace
        target = np.trace(C) / n_channels * np.eye(n_channels)
        Ctrain[i] = (1 - shrinkage) * C + shrinkage * target
    return Ctrain

def riemann(array, CtempX):
    Ctrain = compute_covariances(array)  # shrinkage happens inside now
    if isinstance(CtempX, int):
        Ctemp = mean_covariance(Ctrain)
        feat = tangent_space(Ctrain, Ctemp)
        return [feat, Ctemp]
    else:
        feat = tangent_space(Ctrain, CtempX)
        return [feat, CtempX]

def plot_pca_lda_boundary(X_feat, y, output_path, base_name):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_feat)
    lda_vis = LinearDiscriminantAnalysis()
    lda_vis.fit(X_pca, y)
    x_min, x_max = -6, 6
    y_min, y_max = -6, 6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = lda_vis.predict(grid)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    colors = {0: "tab:green", 1: "tab:brown"}
    labels = {0: "Relax", 1: "Walk MI"}
    for label in [0, 1]:
        idx = y == label
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], color=colors[label], label=labels[label], alpha=0.85, s=60)
    plt.xlabel("PC1", fontsize=16)
    plt.ylabel("PC2", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    output_filename = f"PCA_RG_{base_name}.png"
    full_output_path = os.path.join(output_path, output_filename)
    plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
    plt.show()
    return pca, lda_vis, X_pca
    
def plot_pca_lda_boundary_fbcsp(X_feat, y, output_path, base_name):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_feat)
    lda_vis = LinearDiscriminantAnalysis()
    lda_vis.fit(X_pca, y)
    x_min, x_max = -6, 6
    y_min, y_max = -6, 6
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = lda_vis.predict(grid)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    colors = {0: "tab:green", 1: "tab:brown"}
    labels = {0: "Relax", 1: "Walk MI"}
    for label in [0, 1]:
        idx = y == label
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], color=colors[label], label=labels[label], alpha=0.85, s=60)
    plt.xlabel("PC1", fontsize=16)
    plt.ylabel("PC2", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.tight_layout()
    output_filename = f"PCA_FBCSP_{base_name}.png"   # <-- nombre distinto
    full_output_path = os.path.join(output_path, output_filename)
    plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
    plt.show()
    return pca, lda_vis, X_pca

def load_xdf_streams(filepath, protocol):
    logger.info(f"Loading XDF: {filepath}")
    # Main EEG streams
    main_streams, fileheader = pyxdf.load_xdf(str(filepath),dejitter_timestamps=True, select_streams=[{'name': 'SourceEEG'},{'name': 'SourceData'},{'name': 'LiveAmpSN-054211-0207'},{'name': 'LiveAmpSN-054208-0183'}])
    stream = main_streams[0]
    stream['time_series'] = (stream['time_series'] / 1000000)
    # Marker stream
    marker_stream, _ = pyxdf.load_xdf(str(filepath), dejitter_timestamps=True, select_streams=[{'name': 'TaskOutput'}])
    # Preprocessed stream
    preprocessed_stream, _ = pyxdf.load_xdf(str(filepath),dejitter_timestamps=True, select_streams=[{'name': 'PreprocessedData'}])
    # Feedback stream
    feedback_stream, _ = pyxdf.load_xdf(str(filepath), dejitter_timestamps=True,select_streams=[{'name': 'FeedbackStates'}])
    streams = {"main": stream,  "marker": marker_stream[0], "preprocessed": preprocessed_stream[0], "feedback": feedback_stream[0]}
    # LaplaceData only exists in Protocol I
    if protocol == "ProtocolI":
        laplace_stream, _ = pyxdf.load_xdf(str(filepath), dejitter_timestamps=True, select_streams=[{'name': 'LaplaceData'}])
        streams["laplace"] = laplace_stream[0]
    return streams, fileheader

def calculate_laplace(data):
    ch_name = 'Cz'
    laplace_data = data.copy().pick_channels([ch_name])
    reference_data = data.copy().pick_channels(TARGET_LAPLACE, ordered=False)
    laplace_cz = (reference_data.get_data().sum(axis=0) / len(reference_data.ch_names))
    laplace_data._data -= laplace_cz[np.newaxis, :]
    logger.info(f"Laplacian Cz shape: {laplace_data.get_data().shape}")
    return laplace_data

def prepare_protocol_I(streams):
    stream = streams["main"]
    marker_stream = streams["marker"]
    preprocessed_stream = streams["preprocessed"]
    laplace_stream = streams["laplace"]
    # Convert streams to MNE Raw
    raw, events, event_id = stream2raw(stream, marker_stream=marker_stream, marker_out=marker_out)
    preprocessed_data, _, _ = stream2raw(preprocessed_stream, marker_stream=marker_stream, marker_out=marker_out)
    laplace_dataraw, _, _ = stream2raw(laplace_stream, marker_stream=marker_stream, marker_out=marker_out)
    # Raw Laplace Cz
    laplaceraw_Cz = laplace_dataraw.copy().pick_channels(['µCz'])
    logger.info(f"size preprocessed data: {preprocessed_data._data.shape} ch names: {preprocessed_data.ch_names}")
    preprocessed_32 = preprocessed_data.copy()
    laplace_data = calculate_laplace(preprocessed_32)
    trial_length = float(cast( str, get_parameters_from_xdf_stream( marker_stream).get("cue_length")))
    walk_events = mne.events_from_annotations(preprocessed_32, regexp=classMI)
    relax_events = mne.events_from_annotations(preprocessed_32, regexp=classrest)
    # First and last trial
    all_onsets = sorted([e[0] for e in walk_events[0]] + [e[0] for e in relax_events[0]])
    if not all_onsets:
        raise ValueError("No MI or Relax events found")
    first_start: float = all_onsets[0] / preprocessed_32.info["sfreq"]
    last_end: float = (all_onsets[-1] / preprocessed_32.info["sfreq"] + trial_length)
    # Reference value (RV)
    segment = laplace_dataraw.get_data(  picks=[1], tmin=first_start, tmax=last_end)
    rv = float(segment.mean())
    if not np.isfinite(rv) or rv == 0:
        raise ValueError(f"Invalid RV: {rv}")
    logger.info(f"trial_length {trial_length} laplace data raw {laplace_dataraw.get_data().shape} segment shape: {segment.shape}")
    logger.info(f"segment mean (rv): {rv:.2f} µV")
    logger.info(f"First trial starts at {first_start:.2f}s, last trial ends at {last_end:.2f}")
    # All-channel epochs
    walk_epochs_all = mne.Epochs(preprocessed_32, walk_events[0], tmin=1, tmax=trial_length, event_id=1, baseline=None, preload=True)[:10]
    relax_epochs_all = mne.Epochs(preprocessed_32, relax_events[0], tmin=1, tmax=trial_length, event_id=1, baseline=None, preload=True)[:10]
    logger.info(f"walk epochs all shape: {walk_epochs_all._data.shape}, relax epochs all shape: {relax_epochs_all._data.shape}")
    logger.info(f"epochs all included channels: {walk_epochs_all.ch_names}")
    # Laplacian epochs
    walk_epochs = mne.Epochs(laplace_data, walk_events[0], tmin=1, tmax=trial_length, event_id=1, baseline=None, preload=True)[:10]
    relax_epochs = mne.Epochs(laplace_data, relax_events[0], tmin=1, tmax=trial_length, event_id=1, baseline=None, preload=True)[:10]
    # Raw Laplace epochs for normalized power plot
    walk_events_lap = mne.events_from_annotations(laplaceraw_Cz, regexp=classMI)
    relax_events_lap = mne.events_from_annotations(laplaceraw_Cz, regexp=classrest)
    walk_plot_epochs = mne.Epochs(laplaceraw_Cz, walk_events_lap[0], event_id=1, tmin=1, tmax=trial_length, baseline=None, preload=True)[:10]
    relax_plot_epochs = mne.Epochs(laplaceraw_Cz, relax_events_lap[0], event_id=1, tmin=1, tmax=trial_length, baseline=None, preload=True)[:10]
    logger.info(f"walk epochs laplace shape: {walk_plot_epochs._data.shape}, relax epochs laplace shape: {relax_plot_epochs._data.shape}") 
    picks = mne.pick_channels(  walk_epochs_all.ch_names, include=INCLUDE_CHANNELS,  exclude=[])
    walk_epochs_all.pick(picks=picks)
    relax_epochs_all.pick(picks=picks)    
    return {
        "walk_epochs_all": walk_epochs_all,
        "relax_epochs_all": relax_epochs_all,
        "walk_epochs": walk_epochs,
        "relax_epochs": relax_epochs,
        "walk_plot_epochs": walk_plot_epochs,
        "relax_plot_epochs": relax_plot_epochs,
        "rv": rv,
        "trial_length": trial_length,
        "sampling_freq": walk_epochs_all.info["sfreq"]
    }

def prepare_protocol_II(streams):
    stream = streams["main"]
    marker_stream = streams["marker"]
    preprocessed_stream = streams["preprocessed"]
    # Convert streams to MNE Raw
    raw, events, event_id = stream2raw(stream,marker_stream=marker_stream, marker_out=marker_out)
    preprocessed_data, _, _ = stream2raw(preprocessed_stream, marker_stream=marker_stream, marker_out=marker_out)
    # Sampling frequency
    sampling_freq = raw.info["sfreq"]
    # Trial duration
    trial_length = float(cast(str, get_parameters_from_xdf_stream( marker_stream).get("cue_length")))
    # Filtered raw signal for Laplacian calculation
    filt_data = raw.copy().filter(l_freq=6,h_freq=30,verbose="WARNING")
    laplace_data = calculate_laplace(filt_data)
    # Extract events from raw
    walk_events = mne.events_from_annotations(raw,regexp=classMI)
    relax_events = mne.events_from_annotations(raw,regexp=classrest)
    # Raw epochs
    walk_epochs = mne.Epochs(raw, walk_events[0], tmin=0, tmax=RELAX_LENGTH_SECONDS, baseline=None, preload=True)[:10]
    relax_epochs = mne.Epochs(raw, relax_events[0], tmin=0, tmax=RELAX_LENGTH_SECONDS, baseline=None, preload=True)[:10]
    # Select channels
    picks = mne.pick_channels(walk_epochs.ch_names, include=INCLUDE_CHANNELS, exclude=[])
    walk_epochs_all = walk_epochs.copy().pick(picks=picks)
    relax_epochs_all = relax_epochs.copy().pick(picks=picks)
    # Trial boundaries
    all_onsets = sorted([e[0] for e in walk_events[0]] + [e[0] for e in relax_events[0]])
    if not all_onsets:
        raise ValueError("No MI or Relax events found")
    first_start = (all_onsets[0] / raw.info["sfreq"])
    last_end = (all_onsets[-1] / raw.info["sfreq"] + trial_length)
    # Preprocessed Cz for normalization
    data_prep = preprocessed_data.copy().pick_channels(['µCz'])
    rv = float(data_prep.get_data(tmin=first_start, tmax=last_end).mean())
    if not np.isfinite(rv) or rv == 0:
        raise ValueError(f"Invalid RV: {rv}")
    # Extract preprocessed epochs
    walk_prep_events = mne.events_from_annotations(data_prep, regexp=classMI)
    relax_prep_events = mne.events_from_annotations(data_prep, regexp=classrest)
    walk_plot_epochs = mne.Epochs(data_prep, walk_prep_events[0], tmin=0, tmax=RELAX_LENGTH_SECONDS, baseline=None, preload=True)[:10]
    relax_plot_epochs = mne.Epochs(data_prep, relax_prep_events[0],  tmin=0, tmax=RELAX_LENGTH_SECONDS, baseline=None, preload=True)[:10]
    # Return standardized data
    return {
        "walk_epochs_all": walk_epochs_all,
        "relax_epochs_all": relax_epochs_all,
        "walk_epochs": walk_epochs,
        "relax_epochs": relax_epochs,
        "walk_plot_epochs": walk_plot_epochs,
        "relax_plot_epochs": relax_plot_epochs,
        "rv": rv,
        "trial_length": trial_length,
        "sampling_freq": sampling_freq
    }


def calculate_erd_ers(walk_epochs, relax_epochs):
    fmin = 8
    fmax = 12
    psds_walk = walk_epochs.compute_psd(fmin=fmin, fmax=fmax, method='welch', n_overlap=512, tmin=0, tmax=10).get_data()
    psds_relax = relax_epochs.compute_psd(fmin=fmin, fmax=fmax, method='welch', n_fft=1024, n_overlap=512, tmin=0, tmax=10).get_data()
    # Average across frequencies and trials
    mean_walk = psds_walk.mean(axis=2).mean(axis=0)
    mean_relax = psds_relax.mean(axis=2).mean(axis=0)
    # Valid channels
    valid = ( np.isfinite(mean_walk) & np.isfinite(mean_relax) & (mean_relax > 0))
    erd_ers = np.full(mean_relax.shape, np.nan, dtype=float)
    erd_ers[valid] = ((mean_walk[valid] - mean_relax[valid]) / mean_relax[valid]) * 100
    # Normalize
    max_abs = np.nanmax(np.abs(erd_ers))
    if np.isfinite(max_abs) and max_abs > 0:
        erd_ers_norm = erd_ers / max_abs
    else:
        erd_ers_norm = np.zeros_like(erd_ers)
    logger.info(f"ERD/ERS: {erd_ers}")
    logger.info(f"Normalized ERD/ERS: {erd_ers_norm}")
    return erd_ers, erd_ers_norm, valid

def plot_erd_ers(epochs, erd_ers_norm, valid, output_path, base_name):
    epochs.set_montage('standard_1020', on_missing='ignore')
    erd_plot = np.nan_to_num(erd_ers_norm, nan=0.0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im, cn = mne.viz.plot_topomap(erd_plot, epochs.info, axes=ax, cmap='RdBu_r',  vlim=(-1, 1),contours=0, mask=~valid,mask_params=dict(marker='x',  markerfacecolor='black', markeredgecolor='black', markersize=10), show=False)
    fig.colorbar(im, ax=ax)
    ax.set_title('Normalized Alpha ERD/ERS (8–12 Hz)')
    plt.tight_layout()
    fig.savefig(
        Path(output_path) / f"{base_name}_ERD_ERS.png", dpi=300, bbox_inches='tight')
    plt.show()
    with open(Path(output_path) / f"{base_name}_erd_ers.txt","w",encoding="utf-8") as f:
        f.write(f"ERD/ERS:\n{erd_ers_norm}\n")
        f.write(f"sfreq: {epochs.info['sfreq']}\n")
        f.write("ch_names: " + ", ".join(epochs.info["ch_names"]))

def plot_normalized_power(data, output_path, base_name):
    walk_epochs = data["walk_plot_epochs"]
    relax_epochs = data["relax_plot_epochs"]
    rv = data["rv"]
    # Normalize trials
    norm_walk = (walk_epochs.get_data() / rv) - 1
    norm_relax = (relax_epochs.get_data() / rv) - 1
    # Average trials
    walk_prep = norm_walk.mean(axis=0)
    relax_prep = norm_relax.mean(axis=0)
    sfreq = walk_epochs.info["sfreq"]
    walk_time = np.arange(walk_prep.shape[-1]) / sfreq
    relax_time = np.arange(relax_prep.shape[-1]) / sfreq
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        relax_time,
        relax_prep[0],
        color="tab:green",
        linewidth=2,
        label="Relax"
    )
    ax.plot(
        walk_time,
        walk_prep[0],
        color="tab:brown",
        linewidth=2,
        label="Motor Imagery"
    )
    ax.set_xlabel("Time [s]", fontsize=18)
    ax.set_ylabel("Relative µ power", fontsize=18)
    ax.set_ylim(-0.4, 0.4)
    ax.legend(fontsize=16)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=14)

    fig.savefig(Path(output_path) / f"MIvsRelax_{base_name}.png", dpi=300, bbox_inches='tight')
    plt.show()
    return walk_prep, relax_prep

def calculate_metrics(cm):
    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    specificity = TN / (TN + FP) if TN + FP > 0 else 0.0
    total = TP + TN + FP + FN
    accuracy = ((TP + TN) / total if total > 0 else 0.0)
    po = accuracy
    pe = (((TP + FN) * (TP + FP)) + ((TN + FP) * (TN + FN))) / (total ** 2) if total > 0 else 0.0
    kappa = ((po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "kappa": kappa
    }

def run_rg_classifier(walk_epochs, relax_epochs,output_path, base_name):
    walk_trials = walk_epochs.get_data()
    relax_trials = relax_epochs.get_data()
    X_trials = np.concatenate((walk_trials, relax_trials), axis=0)
    y_trials = np.concatenate((np.ones(len(walk_trials)),np.zeros(len(relax_trials))))
    window = int(WINDOW_SECONDS * walk_epochs.info["sfreq"])
    step = int((1 - WINDOW_OVERLAP) * window)
    # Repeated split
    real = run_repeated_split(X_trials, y_trials, walk_epochs.info, window, step, n_repeats=100, random_state=42, verbose=True, return_first_split_features=True)
    cm = real["confusion_matrix"]
    metrics = calculate_metrics(cm)
    logger.info(f"RG accuracy: {metrics['accuracy']:.3f} RG kappa: {metrics['kappa']:.3f}")
    logger.info(f"RG confusion matrix:\n{cm}")
    #Plot PCA
    plot_pca_lda_boundary(real["first_split_feat_train"], real["first_split_y_train"], output_path=output_path, base_name=base_name)
    # Permutation test
    n_perm = 100
    n_repeats_perm = 20
    rng = np.random.default_rng(0)
    null_acc = np.zeros(n_perm)
    observed = real["acc_trial"].mean()
    for p in range(n_perm):
        y_shuffled = rng.permutation(y_trials)
        perm_result = run_repeated_split(X_trials, y_shuffled, walk_epochs.info, window, step, n_repeats=n_repeats_perm, random_state=p, verbose=False)
        null_acc[p] = perm_result["acc_trial"].mean()
    p_value = ( 1 + np.sum(null_acc >= observed)) / (n_perm + 1)
    metrics["p_value"] = p_value
    metrics["confusion_matrix"] = cm
    metrics["observed_accuracy"] = observed
    logger.info(f"RG permutation p-value: {p_value:.4f}")
    return metrics

def run_fbcsp_classifier(walk_epochs, relax_epochs, frequency_bands=None,n_components_candidates=(2, 4, 6),n_tuning_splits=20, n_cv_splits=50, n_permutations=100, n_permutation_splits=10, test_size=0.20, random_state=42, output_path=None, base_name=None, logger=None,):
    if frequency_bands is None:
        frequency_bands = [ (8, 12),  (12, 16),  (16, 20),  (20, 24), (24, 30)]
    if not np.isclose(walk_epochs.info['sfreq'], relax_epochs.info['sfreq']):
        raise ValueError("Walk and relax sampling frequencies differ.")
    if walk_epochs.ch_names != relax_epochs.ch_names:
        raise ValueError("Walk and relax channel orders differ.")
    if len(walk_epochs.ch_names) != 23:
        raise ValueError(f"Expected 23 channels, got {len(walk_epochs.ch_names)}.")
    walk_trials = walk_epochs.get_data()
    relax_trials = relax_epochs.get_data()
    X_trials = np.concatenate((walk_trials, relax_trials), axis=0)
    y_trials = np.concatenate(( np.ones(len(walk_trials), dtype=int),np.zeros(len(relax_trials), dtype=int)))
    window = int(WINDOW_SECONDS * walk_epochs.info["sfreq"])
    step = int((1 - WINDOW_OVERLAP) * window)
    logger.info(f"window {window} step {step}")
    if window <= 0 or step <= 0:
        raise ValueError("Invalid window length or overlap.")
    if window > X_trials.shape[-1]:
        raise ValueError("Window length exceeds epoch duration.")
    # Select best "n" compoen
    candidate_n = list(n_components_candidates)
    select_cv = StratifiedShuffleSplit(n_splits=n_tuning_splits, test_size=test_size, random_state=random_state)
    n_scores = {n: [] for n in candidate_n}
    for sel_fold, (train_idx, test_idx) in enumerate(select_cv.split(X_trials, y_trials), start=1):
        X_train_trials = X_trials[train_idx]
        y_train_trials = y_trials[train_idx]
        X_test_trials = X_trials[test_idx]
        y_test_trials = y_trials[test_idx]
        train_epochs = mne.EpochsArray(X_train_trials, walk_epochs.info, verbose=False)
        test_epochs = mne.EpochsArray(X_test_trials, walk_epochs.info, verbose=False)
        band_train_windows = []
        band_test_windows = []
        train_trial_ids = None
        test_trial_ids = None
        for band in frequency_bands:
            train_epochs_band = train_epochs.copy().filter( band[0], band[1], verbose=False)
            test_epochs_band = test_epochs.copy().filter( band[0], band[1], verbose=False)
            train_windows, train_trial_ids = sliding_window_epochs2( train_epochs_band, window, step)
            test_windows, test_trial_ids = sliding_window_epochs2(test_epochs_band, window, step)
            band_train_windows.append(train_windows.get_data())
            band_test_windows.append(test_windows.get_data())
        y_tr = y_train_trials[train_trial_ids]
        y_te = y_test_trials[test_trial_ids]

        for n in candidate_n:
            features_train = []
            features_test = []
            for X_tr_band, X_te_band in zip(band_train_windows, band_test_windows):
                csp = CSP(n_components=n, log=True,  norm_trace=True,  reg='ledoit_wolf')
                X_tr_csp = csp.fit_transform(X_tr_band, y_tr)
                X_te_csp = csp.transform(X_te_band)
                features_train.append(X_tr_csp)
                features_test.append(X_te_csp)
            X_train_final = np.concatenate(features_train, axis=1)
            X_test_final = np.concatenate(features_test, axis=1)
            lda = LinearDiscriminantAnalysis(solver='lsqr',  shrinkage='auto')
            lda.fit(X_train_final, y_tr)
            pred = lda.predict(X_test_final)
            n_scores[n].append(accuracy_score(y_te, pred))
    mean_scores = {n: np.mean(scores) for n, scores in n_scores.items()}
    best_n = max(mean_scores, key=mean_scores.get)
    logger.info(f"Selected n_components = {best_n}")
    #Stratified cross-validation
    cv = StratifiedShuffleSplit(n_splits=n_cv_splits, test_size=test_size, random_state=random_state + 1)
    fold_acc = []
    fold_kappa = []
    fold_cm = []
    first_split_feat_train = None
    first_split_y_train = None
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_trials, y_trials), start=1):
        X_train_trials = X_trials[train_idx]
        y_train_trials = y_trials[train_idx]
        X_test_trials = X_trials[test_idx]
        y_test_trials = y_trials[test_idx]
        train_epochs = mne.EpochsArray(X_train_trials, walk_epochs.info, verbose=False)
        test_epochs = mne.EpochsArray(X_test_trials, walk_epochs.info, verbose=False)
        features_train = []
        features_test = []
        for band in frequency_bands:
            train_epochs_band = train_epochs.copy().filter(band[0], band[1], verbose=False)
            test_epochs_band = test_epochs.copy().filter(band[0], band[1], verbose=False)
            train_windows, train_trial_ids = sliding_window_epochs2(train_epochs_band, window, step)
            test_windows, test_trial_ids = sliding_window_epochs2(test_epochs_band, window, step)
            X_tr_band = train_windows.get_data()
            X_te_band = test_windows.get_data()
            y_tr = y_train_trials[train_trial_ids]
            csp = CSP(n_components=best_n, log=True, norm_trace=True, reg='ledoit_wolf')
            X_tr_csp = csp.fit_transform(X_tr_band, y_tr)
            X_te_csp = csp.transform(X_te_band)
            features_train.append(X_tr_csp)
            features_test.append(X_te_csp)
        X_train_final = np.concatenate(features_train, axis=1)
        X_test_final = np.concatenate(features_test, axis=1)
        lda = LinearDiscriminantAnalysis( solver='lsqr', shrinkage='auto')
        lda.fit(X_train_final, y_tr)
        pred = lda.predict(X_test_final)
        if fold == 1:
            first_split_feat_train = X_train_final
            first_split_y_train = y_tr
        # Majority voting at trial level
        trial_pred = []
        trial_true = []
        for t in np.unique(test_trial_ids):
            idx = test_trial_ids == t
            vote = mode( pred[idx], keepdims=False).mode
            trial_pred.append(vote)
            trial_true.append(y_test_trials[t])
        acc = accuracy_score(trial_true, trial_pred)
        kappa = cohen_kappa_score(trial_true, trial_pred)
        cm = confusion_matrix(trial_true, trial_pred,labels=[0, 1])
        fold_acc.append(acc)
        fold_kappa.append(kappa)
        fold_cm.append(cm)
    # Calculate final metrics
    total_cm = np.sum(fold_cm, axis=0)
    TN, FP = total_cm[0, 0], total_cm[0, 1]
    FN, TP = total_cm[1, 0], total_cm[1, 1]
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    specificity = TN / (TN + FP) if TN + FP > 0 else 0.0
    total = TP + TN + FP + FN
    accuracy = ((TP + TN) / total if total > 0 else 0.0)
    po = accuracy
    pe = (((TP + FN) * (TP + FP) + (TN + FP) * (TN + FN)) / total**2 if total > 0 else 0.0)
    kappa = ((po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0)
    real_acc = np.mean(fold_acc)
    logger.info(f"ACC trial level: {np.mean(fold_acc):.3f} "f"± {np.std(fold_acc):.3f}")
    logger.info(f"Kappa: {np.mean(fold_kappa):.3f} " f"± {np.std(fold_kappa):.3f}")
    logger.info(f"Confusion matrix:\n{total_cm}")
    logger.info(f"Accuracy: {accuracy:.3f} " f"Precision: {precision:.3f} " f"Recall: {recall:.3f} " f"Specificity: {specificity:.3f} " f"Kappa: {kappa:.3f}")
    #PCA plot
    if output_path is not None and base_name is not None:
        plot_pca_lda_boundary_fbcsp(first_split_feat_train, first_split_y_train,  output_path=output_path,  base_name=base_name)
    #Permtation test
    perm_cv = StratifiedShuffleSplit(n_splits=n_permutation_splits, test_size=test_size, random_state=random_state + 2)
    perm_acc = []
    rng = np.random.default_rng(random_state)
    for i in range(n_permutations):
        logger.info(f"Permutation {i + 1}/{n_permutations}")
        y_perm_trials = rng.permutation(y_trials)
        split_accs = []
        for train_idx, test_idx in perm_cv.split(X_trials, y_perm_trials):
            X_train_trials = X_trials[train_idx]
            y_train_perm_trials = y_perm_trials[train_idx]
            X_test_trials = X_trials[test_idx]
            y_test_perm_trials = y_perm_trials[test_idx]
            train_epochs = mne.EpochsArray(X_train_trials, walk_epochs.info, verbose=False)
            test_epochs = mne.EpochsArray(X_test_trials,  walk_epochs.info, verbose=False)
            features_train = []
            features_test = []
            for band in frequency_bands:
                train_epochs_band = train_epochs.copy().filter(band[0], band[1], verbose=False)
                test_epochs_band = test_epochs.copy().filter(band[0], band[1], verbose=False)
                train_windows, train_ids = sliding_window_epochs2(train_epochs_band, window, step)
                test_windows, test_ids = sliding_window_epochs2(test_epochs_band, window, step)
                X_tr_band = train_windows.get_data()
                X_te_band = test_windows.get_data()
                y_tr = y_train_perm_trials[train_ids]
                csp = CSP(n_components=best_n, log=True, norm_trace=True, reg='ledoit_wolf')
                X_tr_csp = csp.fit_transform(X_tr_band, y_tr)
                X_te_csp = csp.transform(X_te_band)
                features_train.append(X_tr_csp)
                features_test.append(X_te_csp)
            X_train_fbcsp = np.concatenate(features_train, axis=1)
            X_test_fbcsp = np.concatenate(features_test, axis=1)
            lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            lda.fit(X_train_fbcsp, y_tr)
            pred = lda.predict(X_test_fbcsp)
            trial_pred = []
            trial_true = []
            for t in np.unique(test_ids):
                idx = test_ids == t
                vote = mode( pred[idx], keepdims=False).mode
                trial_pred.append(vote)
                trial_true.append(y_test_perm_trials[t])
            split_accs.append(accuracy_score(trial_true, trial_pred))
        perm_acc.append(np.mean(split_accs))
    perm_acc = np.asarray(perm_acc)
    p_value = (np.sum(perm_acc >= real_acc) + 1) / (n_permutations + 1)
    percentile = (np.sum(perm_acc <= real_acc) / n_permutations) * 100
    logger.info(f"Real ACC: {real_acc:.3f}")
    logger.info(f"Permutation mean: {perm_acc.mean():.3f}")
    logger.info(f"Permutation std: {perm_acc.std():.3f}")
    logger.info(f"p-value: {p_value:.4f}")
    logger.info(f"Percentile: {percentile:.2f}")
    return {
        "accuracy": accuracy,
        "real_accuracy": real_acc,
        "std_accuracy": np.std(fold_acc),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "kappa": kappa,
        "mean_fold_kappa": np.mean(fold_kappa),
        "confusion_matrix": total_cm,
        "fold_accuracies": fold_acc,
        "fold_kappa": fold_kappa,
        "best_n_components": best_n,
        "tuning_results": mean_scores,
        "p_value": p_value,
        "percentile": percentile,
        "permutation_scores": perm_acc,
        "first_split_feat_train": first_split_feat_train,
        "first_split_y_train": first_split_y_train,
        "n_channels": X_trials.shape[1],
        "n_trials": len(X_trials),
        "frequency_bands": frequency_bands,
    }


def process_participant(filepath, protocol, group_type):
    filepath = Path(filepath)
    base_name = filepath.stem
    output_path = filepath.parent
    logger.info("\n" + "=" * 70)
    logger.info(f"Participant: {base_name}")
    logger.info(f"Protocol: {protocol}")
    logger.info(f"Group: {group_type}")
    logger.info("=" * 70)
    # LOAD XDF
    streams, fileheader = load_xdf_streams(filepath, protocol)
    if protocol == "ProtocolI":
        data = prepare_protocol_I(streams)
    elif protocol == "ProtocolII":
        data = prepare_protocol_II(streams)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    # SHARED ANALYSIS
    walk_epochs = data["walk_epochs_all"]
    relax_epochs = data["relax_epochs_all"]
    if ERD_ERS:
        erd_ers, erd_ers_norm, valid = calculate_erd_ers(walk_epochs,relax_epochs)
        plot_erd_ers(walk_epochs, erd_ers_norm, valid,output_path, base_name)
    # Normalized power
    plot_normalized_power(data, output_path, base_name)
    # CLASSIFIERS
    rg_results = run_rg_classifier( walk_epochs, relax_epochs, output_path, base_name)
    fbcsp_results = run_fbcsp_classifier(walk_epochs=walk_epochs,relax_epochs=relax_epochs, output_path=output_path, base_name=base_name, logger=logger)
    results = {
        "participant": base_name,
        "protocol": protocol,
        "group": group_type,

        "accuracy_RG": rg_results["accuracy"],
        "kappa_RG": rg_results["kappa"],
        "precision_RG": rg_results["precision"],
        "recall_RG": rg_results["recall"],
        "specificity_RG": rg_results["specificity"],
        "p_value_RG": rg_results["p_value"],

        "accuracy_FBCSP": fbcsp_results["accuracy"],
        "kappa_FBCSP": fbcsp_results["kappa"],
        "precision_FBCSP": fbcsp_results["precision"],
        "recall_FBCSP": fbcsp_results["recall"],
        "specificity_FBCSP": fbcsp_results["specificity"],
        "p_value_FBCSP": fbcsp_results["p_value"],
        "percentile_FBCSP": fbcsp_results["percentile"],
        "best_CSP_components": fbcsp_results["best_n_components"]
    }
    return results

########################################################################## Main code ##########################################################################
FBCSP = True
Windowing = True
classMI= 'WALK'
classrest= 'RELAX'
marker_out = 2
all_participant_results = []
INPUT_FOLDER = Path(r"A:\ARTICLES\Frontiers2026orMDPI2026\GithubFiles")
MATCHED_PARTICIPANTS = [f"P{i:02d}" for i in range(1, 7)]
INDEPENDENT_PARTICIPANTS = ([f"P{i:02d}" for i in range(7, 14)] + [f"P{i:02d}" for i in range(14, 21)])
RELAX_LENGTH_SECONDS = 10.0

INCLUDE_CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8','FC5', 'FC1', 'FC2', 'FC6','C3', 'Cz', 'C4','CP5', 'CP1', 'CP2', 'CP6','P7', 'P3', 'Pz', 'P4', 'P8']
TARGET_CZ = ['Cz']
TARGET_LAPLACE = ['Fz', 'C3', 'C4', 'Pz', 'F3', 'F4', 'P3', 'P4']
BANDS = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 30)]
CLASSIFIER_TMIN = 0.0
CLASSIFIER_TMAX = 10.0
WINDOW_SECONDS = 2.0
WINDOW_OVERLAP = 0.40
ERD_ERS = True
####################################################################################################################################################
# main execution
participant_groups = {**{ p: "matched" for p in MATCHED_PARTICIPANTS}, **{p: "independent" for p in INDEPENDENT_PARTICIPANTS}}
xdf_files = sorted(INPUT_FOLDER.glob("*.xdf"))
logger.info(f"Found {len(xdf_files)} XDF files")
for filepath in xdf_files:
    match = re.match(r"^(P\d{2})_(ProtocolI|ProtocolII)$", filepath.stem,  re.IGNORECASE)
    if not match:
        logger.warning(f"Skipping: {filepath.name}")
        continue
    participant = match.group(1).upper()
    protocol = match.group(2)
    group_type = participant_groups.get(participant)
    if group_type is None:
        logger.warning(f"Unknown participant: {participant}")
        continue
    try:
        results = process_participant(filepath=filepath,  protocol=protocol,  group_type=group_type)
        all_participant_results.append(results)
    except Exception as error:
        logger.exception(f"Failed processing {filepath.name}: {error}")
