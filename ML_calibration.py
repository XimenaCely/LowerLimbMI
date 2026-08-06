##############################################################Libraries##########################################################################
import sys
import os
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
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import uniform_filter1d
mne.set_log_level(verbose="WARNING")
from misc.log import getLogger
logger = getLogger(__name__)
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.tangentspace import tangent_space
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from mne.decoding import CSP
from mne.filter import filter_data
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import mode

sys.path.append(os.getcwd())
sys.path.append(os.path.abspath('../misc'))
from pathlib import Path

##############################################################Parameters##########################################################################
filepath = f"C://...//P01_ProtocolI.xdf"    # Change "..." for the location in your computer of the protocol files

output_path = os.path.dirname(filepath)
logger.info(f"Output path: {output_path}")
base_name = os.path.splitext(os.path.basename(filepath))[0]
logger.info(f"Base name: {base_name}")

logger.info(f"Selected file: {filepath} type {type(filepath)}")
folder = os.path.dirname(filepath)
logger.info(f"{folder}")

FBCSP = True
Windowing = True
classMI= 'WALK'
classrest= 'RELAX'
marker_out = 2
ERD_ERS = True

#####################################################################Functions##########################################################################

def get_smoothed_envelope(data, smooth_sigma=20):
    analytic_signal = hilbert(data, axis=-1)    
    envelope = np.abs(analytic_signal)    
    envelope_smooth = gaussian_filter1d(envelope, sigma=smooth_sigma, axis=-1)    
    return envelope_smooth

def bandpass_filter(data, low_freq, high_freq, sfreq, order=4):
    nyquist = sfreq / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = np.zeros_like(data)
    for e in range(data.shape[0]):
        for ch in range(data.shape[1]):
            filtered[e, ch, :] = filtfilt(b, a, data[e, ch, :])
    return filtered

def sliding_window_epochs(epochs, window, step):
    data = epochs.get_data()   # shape: (n_epochs, n_channels, n_times)
    X = []    
    for ep in data:
        n_times = ep.shape[1]
        for start in range(0, n_times - window + 1, step):
            X.append(ep[:, start:start+window])
    X = np.array(X)
    info = epochs.info
    new_epochs = mne.EpochsArray(
        X,
        info,
        tmin=0
    )
    return new_epochs

def run_repeated_split(X_trials, y_trials, info, window, step,
                        n_repeats=100, test_size=0.2, random_state=42,
                        verbose=False, return_first_split_features=False):

    sss = StratifiedShuffleSplit( n_splits=n_repeats, test_size=test_size, random_state=random_state )

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
        clf = make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        )
        clf.fit(feat_train[0], y_train)
        pred = clf.predict(feat_test[0])

        if return_first_split_features and split == 1:
            first_split_feat_train = feat_train[0]
            first_split_y_train = y_train
            first_split_feat_test = feat_test[0]
            first_split_y_test = y_test

        acc_window = accuracy_score(y_test, pred)
        kappa_window = cohen_kappa_score(y_test, pred)
        acc_window_all.append(acc_window)
        kappa_window_all.append(kappa_window)

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

    result = {
        "acc_trial": np.array(acc_trial_all),
        "kappa_trial": np.array(kappa_trial_all),
        "acc_window": np.array(acc_window_all),
        "kappa_window": np.array(kappa_window_all),
        "confusion_matrix": np.sum(cm_all, axis=0),
    }

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

            X.append( ep[:, start:start+window] )

            trial_ids.append( trial_idx )

    X = np.array(X)
    info = epochs.info
    new_epochs = mne.EpochsArray( X, info, tmin=0 )
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

def extract_fbcsp_features(X_train, y_train, X_test, bands, n_components, sfreq):

    features_train = []
    features_test = []

    for band in bands:

        X_train_band = filter_data(X_train, sfreq=sfreq, l_freq=band[0], h_freq=band[1], verbose=False)
        X_test_band = filter_data(X_test, sfreq=sfreq, l_freq=band[0], h_freq=band[1], verbose=False )
        csp = CSP(n_components=n_components, log=True, norm_trace=True, reg='ledoit_wolf' )
        X_train_csp = csp.fit_transform( X_train_band, y_train  )
        X_test_csp = csp.transform( X_test_band )
        features_train.append( X_train_csp )
        features_test.append( X_test_csp   )

    X_train_final = np.concatenate( features_train, axis=1  )
    X_test_final = np.concatenate( features_test, axis=1   )
    return X_train_final, X_test_final

def trial_majority_vote( predictions, trial_ids, y_trials):

    trial_pred = []
    trial_true = []

    for t in np.unique(trial_ids):

        idx = trial_ids == t
        vote = mode( predictions[idx], keepdims=False ).mode 
        trial_pred.append( vote)
        trial_true.append( y_trials[t])
    return np.array(trial_true), np.array(trial_pred)

def plot_pca_lda_boundary(X_feat, y, output_path, base_name):
    
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_feat)
    
    lda_vis = LinearDiscriminantAnalysis()
    lda_vis.fit(X_pca, y)

    x_min, x_max = -6, 6
    y_min, y_max = -6, 6

    xx, yy = np.meshgrid( np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300) )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = lda_vis.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)

    colors = {0: "tab:green", 1: "tab:brown"}
    labels = {0: "Relax", 1: "Walk MI"}

    for label in [0, 1]:
        idx = y == label
        plt.scatter( X_pca[idx, 0], X_pca[idx, 1], color=colors[label], label=labels[label], alpha=0.85, s=60 )

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
    # plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
    plt.show()
    return pca, lda_vis, X_pca
    
def plot_pca_lda_boundary_fbcsp(X_feat, y, output_path, base_name):

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_feat)

    lda_vis = LinearDiscriminantAnalysis()
    lda_vis.fit(X_pca, y)

    x_min, x_max = -6, 6
    y_min, y_max = -6, 6

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = lda_vis.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)

    colors = {0: "tab:green", 1: "tab:brown"}
    labels = {0: "Relax", 1: "Walk MI"}

    for label in [0, 1]:
        idx = y == label
        plt.scatter( X_pca[idx, 0], X_pca[idx, 1], color=colors[label], label=labels[label], alpha=0.85, s=60 )

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
    # plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
    plt.show()
    return pca, lda_vis, X_pca

########################################################################## Main code ##########################################################################
stream, fileheader = pyxdf.load_xdf(filepath, dejitter_timestamps=True, select_streams=[{'name': 'SourceEEG'}, {'name': 'SourceData'}, {'name': 'LiveAmpSN-054211-0207'}, {'name': 'LiveAmpSN-054208-0183'}])
stream = stream[0]
stream['time_series'] /= 1000000

marker_stream, _ = pyxdf.load_xdf(filepath, dejitter_timestamps=True, select_streams=[{'name': 'TaskOutput'}])
marker_stream = marker_stream[0]

preprocessed_stream, _ = pyxdf.load_xdf(filepath, dejitter_timestamps=True, select_streams=[{'name': 'PreprocessedData'}])
preprocessed_stream = preprocessed_stream[0]

laplace_stream, _ = pyxdf.load_xdf(filepath, dejitter_timestamps=True, select_streams=[{'name': 'LaplaceData'}])
laplace_stream = laplace_stream[0]

feedback_stream, _ = pyxdf.load_xdf(filepath, dejitter_timestamps=True, select_streams=[{'name': 'FeedbackStates'}])
feedback_stream = feedback_stream[0]

raw, events, event_id = stream2raw(stream, marker_stream=marker_stream, marker_out=marker_out)
preprocessed_data, ev, ev_id = stream2raw(preprocessed_stream,marker_stream=marker_stream, marker_out=marker_out)
laplace_dataraw, ev, ev_id = stream2raw(laplace_stream,marker_stream=marker_stream, marker_out=marker_out)    
laplaceraw_Cz = laplace_dataraw.copy().pick_channels(['µCz'])

# Preprocess data to obtain Laplace in filtered 32 channels
logger.info(f"size preprocessed data: {preprocessed_data._data.shape} ch names: {preprocessed_data.ch_names}")
preprocessed_32 = preprocessed_data.copy()
ch_name = 'Cz'
filt_cz = preprocessed_data.copy().pick_channels([ch_name]).get_data()[0]
times = preprocessed_data.times
targetCz = [ch_name]
targetLaplace = ['Fz','C3','C4','Pz', 'F3', 'F4', 'P3', 'P4']
laplace_data = preprocessed_data.copy().pick_channels(targetCz)
ref_data = preprocessed_data.copy().pick_channels(targetLaplace, ordered=False)
laplaceCz = np.sum(ref_data._data[:], axis=0) * (1/len(ref_data.ch_names)) 
laplace_data._data = laplace_data._data - laplaceCz
logger.info(f"laplace calculated shape: {laplace_data._data.shape} laplace ch names: {laplace_data.ch_names}")

ch_types = ['eeg'] * len(raw.ch_names)
sampling_freq = raw.info['sfreq']       #laplace_dataraw
info = mne.create_info(ch_names=[ch_name], sfreq=sampling_freq, ch_types=['eeg'])  

events, event_id = mne.events_from_annotations(raw)
trial_length: float = float(cast(str, get_parameters_from_xdf_stream(marker_stream).get("cue_length")))
Walk_events = mne.events_from_annotations(preprocessed_32, regexp=classMI) #Motor imagery task
Relax_events = mne.events_from_annotations(preprocessed_32, regexp=classrest) #Relax task
close_onsets = [e[0] for e in Walk_events[0]]
relax_onsets = [e[0] for e in Relax_events[0]]
all_onsets_sorted = sorted(close_onsets + relax_onsets)
first_trial_start_time: float = all_onsets_sorted[0] / preprocessed_32.info["sfreq"]
last_trial_end_time: float = all_onsets_sorted[-1] / preprocessed_32.info["sfreq"] + trial_length        
segment = laplace_dataraw.get_data( picks=[1], tmin=first_trial_start_time, tmax=last_trial_end_time )

# rv calculation
rv = float(segment.mean()) 
logger.info(f"trial_length {trial_length} laplace data raw {laplace_dataraw.get_data().shape} segment shape: {segment.shape}")
logger.info(f"segment mean (rv): {rv:.2f} µV")
logger.info(f"First trial starts at {first_trial_start_time:.2f}s, last trial ends at {last_trial_end_time:.2f}")
##############################################################################################################################################
walk_epochs_all = mne.Epochs (preprocessed_32, Walk_events[0], tmin=1, event_id=1, tmax=trial_length, baseline=None, preload=True)
relax_epochs_all = mne.Epochs (preprocessed_32, Relax_events[0], tmin=1, event_id=1, tmax=trial_length, baseline=None, preload=True)
logger.info(f"walk epochs all shape: {walk_epochs_all._data.shape}, relax epochs all shape: {relax_epochs_all._data.shape}")
##############################################################################################################################################
walk_epochs = mne.Epochs (laplace_data, Walk_events[0], tmin=1, event_id=1, tmax=trial_length, baseline=None, preload=True)
relax_epochs = mne.Epochs (laplace_data, Relax_events[0], tmin=1, event_id=1, tmax=trial_length, baseline=None, preload=True)
########################### Laplace raw Epochs ########################################
walk_events_rescaled = mne.events_from_annotations(laplaceraw_Cz, regexp=classMI)
relax_events_rescaled = mne.events_from_annotations(laplaceraw_Cz, regexp=classrest)  
walk_epochslap = mne.Epochs(laplaceraw_Cz, walk_events_rescaled[0], event_id=1, tmin=1, tmax=trial_length, baseline=None, preload=True)
relax_epochslap = mne.Epochs(laplaceraw_Cz, relax_events_rescaled[0], event_id=1, tmin=1, tmax=trial_length, baseline=None, preload=True)    
exclude = ['FT9', 'TP9', 'FT10', 'TP10', 'O1', 'O2', 'T7', 'T8', 'POz']
picks = mne.pick_channels(walk_epochs_all.info['ch_names'], include=[], exclude=exclude)
walk_epochs_all = walk_epochs_all.pick(picks=picks)
relax_epochs_all = relax_epochs_all.pick(picks=picks)

walk_trials = walk_epochs_all.get_data()
relax_trials = relax_epochs_all.get_data()

logger.info(f"Walk trials: {walk_trials.shape}")
logger.info(f"Relax trials: {relax_trials.shape}")

# Split por trials
walk_train, walk_val = train_test_split( walk_trials, test_size=0.2,  random_state=42, shuffle=True)
relax_train, relax_val = train_test_split( relax_trials, test_size=0.2, random_state=42, shuffle=True)


walk_train_epochs = mne.EpochsArray( walk_train, info=walk_epochs_all.info)

walk_val_epochs = mne.EpochsArray( walk_val, info=walk_epochs_all.info )

relax_train_epochs = mne.EpochsArray( relax_train, info=relax_epochs_all.info  )

relax_val_epochs = mne.EpochsArray( relax_val, info=relax_epochs_all.info )

window = int(2 * sampling_freq)
step = int(0.6 * window)

walk_train_windows = sliding_window_epochs( walk_train_epochs, window, step)

walk_val_windows = sliding_window_epochs( walk_val_epochs, window, step )

relax_train_windows = sliding_window_epochs( relax_train_epochs, window, step)

relax_val_windows = sliding_window_epochs( relax_val_epochs, window, step )


walk_train_data = walk_train_windows.get_data()
walk_val_data = walk_val_windows.get_data()

relax_train_data = relax_train_windows.get_data()
relax_val_data = relax_val_windows.get_data()

X_train = np.concatenate( (walk_train_data, relax_train_data), axis=0 )

y_train = np.concatenate( (np.ones(len(walk_train_data)), np.zeros(len(relax_train_data)) ))

X_val = np.concatenate( (walk_val_data, relax_val_data), axis=0 )

y_val = np.concatenate( ( np.ones(len(walk_val_data)), np.zeros(len(relax_val_data)) ))

logger.info(f"X_train: {X_train.shape}")
logger.info(f"X_val: {X_val.shape}")
logger.info(f"y_train: {y_train.shape}")
logger.info(f"y_val: {y_val.shape}")

############################################################################################################## RG ##################################################################################################################
walk_trials = walk_epochs_all.get_data()
relax_trials = relax_epochs_all.get_data()

X_trials = np.concatenate((walk_trials, relax_trials), axis=0)
y_trials = np.concatenate((np.ones(len(walk_trials)), np.zeros(len(relax_trials))))

window = int(2 * sampling_freq)
step = int(0.6 * window)  # 40% overlap, same params you already had

# ---- 1. Real result ----
real = run_repeated_split(
    X_trials, y_trials, walk_epochs_all.info,
    window, step, n_repeats=100, random_state=42, verbose=True, return_first_split_features=True 
)

logger.info( f"Real Trial ACC RG: {real['acc_trial'].mean():.3f} ± {real['acc_trial'].std():.3f}" )
logger.info(f"Confusion matrix (summed over splits):\n{real['confusion_matrix']}")
cm = real['confusion_matrix']

TN, FP = cm[0, 0], cm[0, 1]
FN, TP = cm[1, 0], cm[1, 1]

precisionRG = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recallRG    = TP / (TP + FN) if (TP + FN) > 0 else 0.0     # = sensitivity
specificityRG = TN / (TN + FP) if (TN + FP) > 0 else 0.0

accuracyRG = (TP + TN) / (TP + TN + FP + FN)

total = TP + TN + FP + FN
po = (TP + TN) / total  # observed agreement

pe = ( ((TP + FN) * (TP + FP)) + ((TN + FP) * (TN + FN)) ) / (total ** 2)  # expected agreement by chance

kappaRG = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0

logger.info(f"Accuracy:    {accuracyRG:.3f}")
logger.info(f"Precision:   {precisionRG:.3f}")
logger.info(f"Recall (Sens): {recallRG:.3f}")
logger.info(f"Specificity: {specificityRG:.3f}")
logger.info(f"Kappa:       {kappaRG:.3f}")

plot_pca_lda_boundary(
    real["first_split_feat_train"],
    real["first_split_y_train"],
    output_path=output_path,
    base_name=base_name
)

########################################################################################## Permutation test #############################################################################################################
n_perm = 100          
n_repeats_perm = 20   
rng = np.random.default_rng(0)

null_acc = np.zeros(n_perm)

for p in range(n_perm):
    y_shuffled = rng.permutation(y_trials)  # shuffle trial labels, keeps class balance

    perm_result = run_repeated_split( X_trials, y_shuffled, walk_epochs_all.info, window, step, n_repeats=n_repeats_perm, random_state=p, verbose=False)
    null_acc[p] = perm_result["acc_trial"].mean()

    if (p + 1) % 20 == 0:
        logger.info(f"Permutation {p+1}/{n_perm} done")

observed = real["acc_trial"].mean()
p_valueRG = (1 + np.sum(null_acc >= observed)) / (n_perm + 1)

logger.info(f"Null distribution: {null_acc.mean():.3f} ± {null_acc.std():.3f}")
logger.info(f"Observed ACC: {observed:.3f} | Permutation p-value: {p_valueRG:.4f}")

###################################################################################Classifier FBCSP+LDA####################################################################################################################################################

if FBCSP:

    X_trials = np.concatenate((walk_trials, relax_trials), axis=0)
    y_trials = np.concatenate( (np.ones(len(walk_trials)), np.zeros(len(relax_trials))) )

    bands = [
        (8, 12),
        (12, 16),
        (16, 20),
        (20, 24),
        (24, 30)
    ]

    sfreq = walk_epochs.info['sfreq']

    candidate_n = [2, 4, 6]

    select_cv = StratifiedShuffleSplit( n_splits=20, test_size=0.2, random_state=0 )

    n_scores = {n: [] for n in candidate_n}

    for sel_fold, (train_idx, test_idx) in enumerate( select_cv.split(X_trials, y_trials), start=1  ):

        X_train_trials = X_trials[train_idx]
        y_train_trials = y_trials[train_idx]

        X_test_trials = X_trials[test_idx]
        y_test_trials = y_trials[test_idx]

        train_epochs = mne.EpochsArray( X_train_trials, walk_epochs_all.info, verbose=False  )

        test_epochs = mne.EpochsArray( X_test_trials, walk_epochs_all.info, verbose=False   )

        # filter + window once per fold, reuse across candidate n
        band_train_windows = []
        band_test_windows = []
        train_trial_ids = test_trial_ids = None

        for band in bands:

            train_epochs_band = train_epochs.copy().filter( band[0], band[1], verbose=False  )
            test_epochs_band = test_epochs.copy().filter( band[0], band[1], verbose=False    )

            train_windows, train_trial_ids = sliding_window_epochs2( train_epochs_band, window, step   )
            test_windows, test_trial_ids = sliding_window_epochs2( test_epochs_band, window, step    )

            band_train_windows.append(train_windows.get_data())
            band_test_windows.append(test_windows.get_data())

        y_tr = y_train_trials[train_trial_ids]
        y_te = y_test_trials[test_trial_ids]

        for n in candidate_n:

            features_train = []
            features_test = []

            for X_tr_band, X_te_band in zip(band_train_windows, band_test_windows):

                csp = CSP(
                    n_components=n,
                    log=True,
                    norm_trace=True,
                    reg='ledoit_wolf'
                )

                X_tr_csp = csp.fit_transform(X_tr_band, y_tr)
                X_te_csp = csp.transform(X_te_band)

                features_train.append(X_tr_csp)
                features_test.append(X_te_csp)

            X_train_final = np.concatenate(features_train, axis=1)
            X_test_final = np.concatenate(features_test, axis=1)

            lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            lda.fit(X_train_final, y_tr)
            pred = lda.predict(X_test_final)

            # window-level accuracy is fine here — this is only for selection
            n_scores[n].append(accuracy_score(y_te, pred))

        # logger.info(f"Selection fold {sel_fold} done")

    mean_scores = {n: np.mean(scores) for n, scores in n_scores.items()}

    best_n = max(mean_scores, key=mean_scores.get)

    logger.info(f"Selected n_components = {best_n}")

    cv = StratifiedShuffleSplit( n_splits=50, test_size=0.2, random_state=42 )

    fold_acc = []
    fold_kappa = []
    fold_cm = []

    first_split_feat_train = None 
    first_split_y_train = None

    for fold, (train_idx, test_idx) in enumerate( cv.split(X_trials, y_trials), start=1 ):

        X_train_trials = X_trials[train_idx]
        y_train_trials = y_trials[train_idx]

        X_test_trials = X_trials[test_idx]
        y_test_trials = y_trials[test_idx]

        train_epochs = mne.EpochsArray( X_train_trials, walk_epochs_all.info, verbose=False  )

        test_epochs = mne.EpochsArray( X_test_trials, walk_epochs_all.info, verbose=False )

        features_train = []
        features_test = []

        for band in bands:

            train_epochs_band = train_epochs.copy().filter( band[0], band[1], verbose=False  )
            test_epochs_band = test_epochs.copy().filter( band[0], band[1], verbose=False  )

            train_windows, train_trial_ids = sliding_window_epochs2( train_epochs_band, window, step )
            test_windows, test_trial_ids = sliding_window_epochs2( test_epochs_band, window, step )

            X_tr_band = train_windows.get_data()
            X_te_band = test_windows.get_data()

            y_tr = y_train_trials[train_trial_ids]

            csp = CSP( n_components=best_n, log=True, norm_trace=True, reg='ledoit_wolf' )

            X_tr_csp = csp.fit_transform(X_tr_band, y_tr)
            X_te_csp = csp.transform(X_te_band)

            features_train.append(X_tr_csp)
            features_test.append(X_te_csp)

        X_train_final = np.concatenate(features_train, axis=1)
        X_test_final = np.concatenate(features_test, axis=1)

        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        lda.fit(X_train_final, y_tr)
        pred = lda.predict(X_test_final)

        if fold == 1:                  # pca plot
            first_split_feat_train = X_train_final  
            first_split_y_train = y_tr 

        trial_pred = []
        trial_true = []

        for t in np.unique(test_trial_ids):
            idx = test_trial_ids == t
            vote = mode(pred[idx], keepdims=False).mode
            trial_pred.append(vote)
            trial_true.append(y_test_trials[t])

        acc = accuracy_score(trial_true, trial_pred)
        kappa = cohen_kappa_score(trial_true, trial_pred)
        cm = confusion_matrix(trial_true, trial_pred, labels=[0, 1])

        fold_acc.append(acc)
        fold_kappa.append(kappa)
        fold_cm.append(cm)
        # logger.info(f"Fold {fold} | ACC={acc:.3f} | Kappa={kappa:.3f}")

    logger.info( f"ACC trial level: {np.mean(fold_acc):.3f} ± {np.std(fold_acc):.3f}"   )
    logger.info(f"acc_all: {fold_acc}")
    logger.info( f"Kappa: {np.mean(fold_kappa):.3f} ± {np.std(fold_kappa):.3f}"  )
    total_cm = np.sum(fold_cm, axis=0)   # <-- nuevo
    logger.info("Confusion matrix (summed over folds):")
    logger.info(f"{total_cm}")

    TN, FP = total_cm[0, 0], total_cm[0, 1]
    FN, TP = total_cm[1, 0], total_cm[1, 1]

    precisionCSP = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recallCSP    = TP / (TP + FN) if (TP + FN) > 0 else 0.0     
    specificityCSP = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    accuracyCSP = (TP + TN) / (TP + TN + FP + FN)

    total = TP + TN + FP + FN
    po = (TP + TN) / total  # observed agreement
    pe = ( ((TP + FN) * (TP + FP)) + ((TN + FP) * (TN + FN)) ) / (total ** 2)  # expected agreement by chance
    kappaCSP = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0

    logger.info(f"Accuracy:    {accuracyCSP:.3f}")
    logger.info(f"Precision:   {precisionCSP:.3f}")
    logger.info(f"Recall (Sens): {recallCSP:.3f}")
    logger.info(f"Specificity: {specificityCSP:.3f}")
    logger.info(f"Kappa:       {kappaCSP:.3f}")

    real_acc = np.mean(fold_acc)

    plot_pca_lda_boundary_fbcsp( first_split_feat_train, first_split_y_train, output_path=output_path,  base_name=base_name   )

    ####################################################################################### Permutation test ####################################################
    logger.info(f"Now, permutation test")

    n_perm = 100            # bump from 100 for finer p-value resolution
    n_perm_splits = 10       # splits averaged per permutation, to match real_acc's variance

    perm_cv = StratifiedShuffleSplit( n_splits=n_perm_splits, test_size=0.2, random_state=42  )

    perm_acc = []
    rng = np.random.default_rng(42)

    for i in range(n_perm):

        logger.info(f"Permutation {i+1}/{n_perm}")
        y_perm_trials = rng.permutation(y_trials)
        split_accs = []

        for train_idx, test_idx in perm_cv.split(X_trials, y_perm_trials):

            X_train_trials = X_trials[train_idx]
            y_train_perm_trials = y_perm_trials[train_idx]

            X_test_trials = X_trials[test_idx]
            y_test_perm_trials = y_perm_trials[test_idx]

            train_epochs = mne.EpochsArray( X_train_trials, walk_epochs_all.info, verbose=False  )

            test_epochs = mne.EpochsArray( X_test_trials, walk_epochs_all.info, verbose=False )

            features_train = []
            features_test = []

            for band in bands:

                # filter BEFORE windowing 
                train_epochs_band = train_epochs.copy().filter( band[0], band[1], verbose=False )
                test_epochs_band = test_epochs.copy().filter( band[0], band[1], verbose=False )

                train_windows, train_ids = sliding_window_epochs2( train_epochs_band, window, step  )
                test_windows, test_ids = sliding_window_epochs2( test_epochs_band, window, step  )

                X_tr_band = train_windows.get_data()
                X_te_band = test_windows.get_data()

                y_tr = y_train_perm_trials[train_ids]
                  # from nested-CV selection
                csp = CSP( n_components=best_n, log=True, norm_trace=True, reg='ledoit_wolf' )

                X_tr_csp = csp.fit_transform(X_tr_band, y_tr)
                X_te_csp = csp.transform(X_te_band)

                features_train.append(X_tr_csp)
                features_test.append(X_te_csp)

            X_train_fbcsp = np.concatenate(features_train, axis=1)
            X_test_fbcsp = np.concatenate(features_test, axis=1)

            lda = LinearDiscriminantAnalysis( solver='lsqr', shrinkage='auto'  )
            lda.fit(X_train_fbcsp, y_tr)
            pred = lda.predict(X_test_fbcsp)

            trial_pred = []
            trial_true = []

            for t in np.unique(test_ids):
                idx = test_ids == t
                vote = mode(pred[idx], keepdims=False).mode
                trial_pred.append(vote)
                trial_true.append(y_test_perm_trials[t])

            split_accs.append( accuracy_score(trial_true, trial_pred) )

        # average over the splits, same aggregation as real_acc
        perm_acc.append(np.mean(split_accs))

    perm_acc = np.array(perm_acc)
    p_valueCSP = ( np.sum(perm_acc >= real_acc) + 1  ) / (  n_perm + 1  )
    percentileCSP = ( np.sum(perm_acc <= real_acc) / n_perm ) * 100

    logger.info("====================")
    logger.info("Permutation test")
    logger.info("====================")
    logger.info(f"Real ACC: {real_acc:.3f}")
    logger.info(f"Permutation mean: {perm_acc.mean():.3f}")
    logger.info(f"Permutation std: {perm_acc.std():.3f}")
    logger.info(f"p-value: {p_valueCSP:.4f}")
    logger.info(f"Percentile: {percentileCSP:.2f}")

txt_filename = f"metrics_FBCSP_RG_{base_name}.txt"
txt_full_path = os.path.join(output_path, txt_filename)

# Save file
# with open(txt_full_path, "w") as f:
#     f.write(f"cm (RG + LDA):\n{real['confusion_matrix']}\n\n")
#     f.write(f"AccuracyRG: {accuracyRG}\n")
#     f.write(f"kappaRG: {kappaRG}\n")
#     f.write(f"PrecisionRG: {precisionRG}\n")
#     f.write(f"RecallRG: {recallRG}\n")
#     f.write(f"SpecificityRG: {specificityRG}\n")
#     f.write(f"p_value RG: {p_valueRG:.4f}\n")

#     f.write(f"cm (FBCSP + LDA):\n{total_cm}\n\n")
#     f.write(f"Validation ACC (FBCSP + LDA): {accuracyCSP}\n")
#     f.write(f"Validation Kappa (FBCSP + LDA): {kappaCSP}\n")
#     f.write(f"Validation Precision (FBCSP + LDA): {precisionCSP}\n")
#     f.write(f"Validation Recall (FBCSP + LDA): {recallCSP}\n")
#     f.write(f"Validation Specificity (FBCSP + LDA): {specificityCSP}\n")
#     f.write(f"Permutation test p-value: {p_valueCSP:.4f}\n")
#     f.write(f"Permutation test percentile: {percentileCSP:.2f}\n") 