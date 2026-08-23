import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
import os

np.random.seed(42)

print("STAGE 3: CONFORMAL PREDICTION FOR MULTIMODAL ICU MORTALITY MODEL")

ALPHA = 0.10  # target miscoverage rate -> 90% nominal coverage

# Custom layer must be defined/imported before load_model, matching Stage 2.
@tf.keras.utils.register_keras_serializable(package="fusion_transformer")
class FeatureGather(layers.Layer):
    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = list(indices)

    def call(self, inputs):
        return tf.gather(inputs, self.indices, axis=2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], len(self.indices))

    def get_config(self):
        config = super().get_config()
        config.update({"indices": self.indices})
        return config


print("\n[1/6] Loading model, calibrator, and data...")

REQUIRED_FILES = {
    '/kaggle/working/models/fusion_transformer.keras': 'Stage 2',
    '/kaggle/working/models/feature_scaler.pkl':       'Stage 2',
    '/kaggle/working/split_indices.npz':               'Stage 2',
    '/kaggle/working/X.npy':                           'Stage 1',
    '/kaggle/working/y.npy':                           'Stage 1',
    '/kaggle/working/feature_names.npy':               'Stage 1',
    '/kaggle/working/subject_id.npy':                  'Stage 1',
    '/kaggle/working/hadm_id.npy':                      'Stage 1',
    '/kaggle/working/observed_mask.npy':               'Stage 1',
}
missing = {f: stage for f, stage in REQUIRED_FILES.items() if not os.path.exists(f)}
if missing:
    print("ERROR: required input files are missing. This is almost always because "
          "/kaggle/working/ was reset (new session/kernel restart) and Stage 1 and/or "
          "Stage 2 haven't been re-run yet IN THIS SESSION -- their outputs don't persist "
          "across separate Kaggle sessions unless explicitly saved as a dataset/output.")
    for f, stage in sorted(missing.items(), key=lambda kv: kv[1]):
        print(f"  MISSING ({stage} output): {f}")
    needed_stages = sorted(set(missing.values()))
    print(f"\nFix: re-run {' then '.join(needed_stages)} in this session, then re-run Stage 3.")
    raise FileNotFoundError(
        f"{len(missing)} required file(s) missing -- see list above. "
        f"Re-run: {' -> '.join(needed_stages)} -> Stage 3."
    )

model = tf.keras.models.load_model('/kaggle/working/models/fusion_transformer.keras')
scaler = joblib.load('/kaggle/working/models/feature_scaler.pkl')

calib_path_iso   = '/kaggle/working/models/calibrator_isotonic.pkl'
calib_path_platt = '/kaggle/working/models/calibrator_platt.pkl'
if os.path.exists(calib_path_iso):
    calibrator = joblib.load(calib_path_iso)
elif os.path.exists(calib_path_platt):
    calibrator = joblib.load(calib_path_platt)
else:
    calibrator = None
    print("No calibrator found on disk; using raw model probabilities uncalibrated.")

X_raw = np.load('/kaggle/working/X.npy')  # unscaled features from Stage 1
y = np.load('/kaggle/working/y.npy')
feature_names = np.load('/kaggle/working/feature_names.npy', allow_pickle=True).tolist()
subject_id = np.load('/kaggle/working/subject_id.npy')
hadm_id = np.load('/kaggle/working/hadm_id.npy')
observed_mask = np.load('/kaggle/working/observed_mask.npy')  # (patients, hours, features), True = real observation
split = np.load('/kaggle/working/split_indices.npz')
train_idx, calib_idx, test_idx = split['train_idx'], split['calib_idx'], split['test_idx']

n_patients, n_hours, n_features = X_raw.shape
print(f"X: {X_raw.shape}  |  y: {y.shape}  |  splits -> train:{len(train_idx)} calib:{len(calib_idx)} test:{len(test_idx)}")

VITALS_IDX = [i for i, f in enumerate(feature_names) if f in
              ['HR', 'RR', 'SBP', 'DBP', 'MBP', 'SpO2', 'TempC', 'Glucose', 'FiO2']]
LABS_IDX   = [i for i, f in enumerate(feature_names) if f in
              ['Creatinine', 'Lactate', 'Hgb', 'WBC', 'Sodium', 'Potassium', 'Bilirubin']]

# Stage 2 fits StandardScaler on the full dataset before splitting; does not
# affect conformal validity since the same transform applies uniformly.
X_2d = X_raw.reshape(-1, n_features)
X_scaled = scaler.transform(X_2d).reshape(n_patients, n_hours, n_features).astype('float32')


# [2/6] Patient-level exchangeability check ("temporal correlation" risk)
print("\n[2/6] Checking patient-level exchangeability across splits...")

def subjects_in(idx):
    return set(subject_id[idx].tolist())

train_subj, calib_subj, test_subj = subjects_in(train_idx), subjects_in(calib_idx), subjects_in(test_idx)
leak_calib_test  = calib_subj & test_subj
leak_train_calib = train_subj & calib_subj
leak_train_test  = train_subj & test_subj

print(f"Unique subjects -> train:{len(train_subj)} calib:{len(calib_subj)} test:{len(test_subj)}")
print(f"Subjects appearing in BOTH calib and test: {len(leak_calib_test)}")
print(f"Subjects appearing in BOTH train and calib: {len(leak_train_calib)}")
print(f"Subjects appearing in BOTH train and test:  {len(leak_train_test)}")

exchangeability_violated = len(leak_calib_test) > 0
if exchangeability_violated:
    print("WARNING: the same patient appears in both the calibration and test sets. This "
          "violates the exchangeability assumption split conformal prediction relies on for "
          "its coverage guarantee -- results below may not carry the claimed guarantee. "
          "Re-run with a patient-grouped split (see fix_split_by_patient() below) for a valid comparison.")
else:
    print("OK: no patient appears in both calibration and test sets. Exchangeability assumption "
          "is not violated by cross-set patient repeats (does not rule out other violations, "
          "e.g. distribution shift over time).")


def fix_split_by_patient(subject_id, y, test_size=0.2, calib_size=0.2, random_state=42):
    """Re-split at the patient level (GroupShuffleSplit-style) so no subject_id crosses
    train/calib/test boundaries. Returns index arrays into the original arrays."""
    from sklearn.model_selection import GroupShuffleSplit
    rng = np.random.RandomState(random_state)
    unique_subjects = np.unique(subject_id)
    # stratify groups roughly by whether they contain any positive outcome, since
    # GroupShuffleSplit doesn't support stratification directly
    subj_to_label = {}
    for s in unique_subjects:
        subj_to_label[s] = y[subject_id == s].max()
    labels_per_subject = np.array([subj_to_label[s] for s in unique_subjects])

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_calib_subj_idx, test_subj_idx = next(gss1.split(unique_subjects, labels_per_subject, groups=unique_subjects))
    train_calib_subjects = unique_subjects[train_calib_subj_idx]
    test_subjects = unique_subjects[test_subj_idx]

    remaining_labels = labels_per_subject[train_calib_subj_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=calib_size / (1 - test_size), random_state=random_state)
    train_subj_idx, calib_subj_idx = next(gss2.split(train_calib_subjects, remaining_labels, groups=train_calib_subjects))
    train_subjects = train_calib_subjects[train_subj_idx]
    calib_subjects = train_calib_subjects[calib_subj_idx]

    train_idx_new = np.where(np.isin(subject_id, train_subjects))[0]
    calib_idx_new = np.where(np.isin(subject_id, calib_subjects))[0]
    test_idx_new  = np.where(np.isin(subject_id, test_subjects))[0]
    return train_idx_new, calib_idx_new, test_idx_new


if exchangeability_violated:
    print("\nRe-splitting by unique patient to remove the exchangeability violation...")
    train_idx_fixed, calib_idx_fixed, test_idx_fixed = fix_split_by_patient(subject_id, y)
    print(f"Patient-grouped split -> train:{len(train_idx_fixed)} calib:{len(calib_idx_fixed)} test:{len(test_idx_fixed)}")
    assert len(subjects_in(calib_idx_fixed) & subjects_in(test_idx_fixed)) == 0, \
        "Patient-grouped split still has cross-set leakage -- this would be a bug."
    print("Verified: patient-grouped split has zero calib/test subject overlap.")
    print("All downstream conformal analysis uses this corrected split, not the original one.")
    # Only calibration/test are corrected here; the model itself was trained on
    # the original split (retraining is out of scope for this stage).
    calib_idx, test_idx = calib_idx_fixed, test_idx_fixed
else:
    calib_idx_fixed, test_idx_fixed = calib_idx, test_idx

# Persist whichever split was actually used downstream, so other scripts (e.g. an LSTM
# baseline meant for a fair, apples-to-apples comparison) can load the identical split
# rather than risk a subtle mismatch from recomputing it independently.
np.savez('/kaggle/working/corrected_split_indices.npz',
         train_idx=train_idx, calib_idx=calib_idx, test_idx=test_idx,
         was_corrected=exchangeability_violated)


# [3/6] Model predictions (calibrated probabilities) for calib and test sets
print("\n[3/6] Generating calibrated probabilities for calib/test sets...")

def predict_calibrated(idx):
    raw = model.predict(X_scaled[idx], verbose=0).flatten()
    if calibrator is not None:
        return calibrator.predict(raw) if hasattr(calibrator, 'predict') and not hasattr(calibrator, 'predict_proba') \
            else calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
    return raw

p_calib = predict_calibrated(calib_idx)
p_test  = predict_calibrated(test_idx)
y_calib_arr = y[calib_idx]
y_test_arr  = y[test_idx]

print(f"Calib probs range: [{p_calib.min():.3f}, {p_calib.max():.3f}]  mean={p_calib.mean():.3f}")
print(f"Test  probs range: [{p_test.min():.3f}, {p_test.max():.3f}]  mean={p_test.mean():.3f}")

# [3b/6] Classification metrics recomputed on the corrected patient-grouped split
# (Stage 2's own printed AUC/F1 use the original stay-level split, different
# patient membership -- not directly comparable to Stage 3/2b's numbers).
print("\n[3b/6] Recomputing classification metrics on the corrected split "
      "(for a fair comparison against the LSTM baseline and against Stage 2's own numbers)...")

from sklearn.metrics import roc_auc_score as _roc_auc_score, f1_score as _f1_score, \
    accuracy_score as _accuracy_score, precision_score as _precision_score, recall_score as _recall_score
from sklearn.metrics import precision_recall_curve as _precision_recall_curve

_prec, _rec, _thr = _precision_recall_curve(y_calib_arr, p_calib)
_f1s = np.divide(2*_prec*_rec, _prec+_rec, out=np.zeros_like(_prec), where=(_prec+_rec) != 0)
_threshold_corrected_split = _thr[np.argmax(_f1s[:-1])]

_auc_c = _roc_auc_score(y_test_arr, p_test)
_f1_c = _f1_score(y_test_arr, (p_test > _threshold_corrected_split).astype(int), zero_division=0)
_acc_c = _accuracy_score(y_test_arr, (p_test > _threshold_corrected_split).astype(int))
_prec_c = _precision_score(y_test_arr, (p_test > _threshold_corrected_split).astype(int), zero_division=0)
_rec_c = _recall_score(y_test_arr, (p_test > _threshold_corrected_split).astype(int), zero_division=0)

print(f"  AUC (corrected split):       {_auc_c:.4f}   (Stage 2's own printed number used a "
      f"different, stay-level test set -- not directly comparable)")
print(f"  F1 (corrected split):        {_f1_c:.4f}")
print(f"  Accuracy (corrected split):  {_acc_c:.4f}")
print(f"  Precision (corrected split): {_prec_c:.4f}")
print(f"  Recall (corrected split):    {_rec_c:.4f}")
print(f"  Threshold (F1-optimal on corrected calib): {_threshold_corrected_split:.4f}")
print("  USE THESE NUMBERS, not Stage 2's console output, when building the LSTM-vs-"
      "Transformer comparison table -- both models are now evaluated on identical test "
      "set membership.")


# [4/6] Split conformal prediction (LAC score) for binary classification
print(f"\n[4/6] Building split-conformal prediction sets (target coverage = {1 - ALPHA:.0%})...")

def conformal_binary(p_calib, y_calib, p_test, alpha):
    """Split conformal prediction for binary classification using the LAC
    (least ambiguous set-valued classifier) nonconformity score:
        s(x, y) = 1 - p_hat(y | x)
    Returns: qhat, prediction_sets (n_test x 2 boolean array: [includes 0, includes 1])
    """
    n = len(y_calib)
    scores = 1 - np.where(y_calib == 1, p_calib, 1 - p_calib)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    qhat = np.quantile(scores, q_level, method='higher')

    score_y0 = 1 - (1 - p_test)  # = p_test  (score if candidate label were 0)
    score_y1 = 1 - p_test        # score if candidate label were 1
    include_0 = score_y0 <= qhat
    include_1 = score_y1 <= qhat
    pred_sets = np.stack([include_0, include_1], axis=1)
    return qhat, pred_sets


qhat, pred_sets = conformal_binary(p_calib, y_calib_arr, p_test, ALPHA)
set_sizes = pred_sets.sum(axis=1)
covered = pred_sets[np.arange(len(y_test_arr)), y_test_arr.astype(int)]

print(f"qhat: {qhat:.4f}")
print(f"Marginal coverage on test: {covered.mean():.4f}  (target: {1 - ALPHA:.4f})")
print(f"Mean prediction set size:  {set_sizes.mean():.4f}")
print(f"Set size distribution -> empty:{(set_sizes==0).mean():.2%}  "
      f"singleton:{(set_sizes==1).mean():.2%}  both-classes:{(set_sizes==2).mean():.2%}")


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (max(0, center - margin), min(1, center + margin))


ci_lo, ci_hi = wilson_ci(covered.sum(), len(covered))
print(f"95% Wilson CI on marginal coverage: [{ci_lo:.4f}, {ci_hi:.4f}]")


# [5/6] Conditional coverage under modality missingness
print("\n[5/6] Conditional coverage by real (pre-imputation) modality missingness...")

vitals_missing_frac = 1 - observed_mask[:, :, VITALS_IDX].mean(axis=(1, 2))
labs_missing_frac    = 1 - observed_mask[:, :, LABS_IDX].mean(axis=(1, 2))

vitals_missing_test = vitals_missing_frac[test_idx]
labs_missing_test   = labs_missing_frac[test_idx]

labs_median = np.median(labs_missing_test)
vitals_median = np.median(vitals_missing_test)

groups = {
    'low labs-missingness (<= median)':  labs_missing_test <= labs_median,
    'high labs-missingness (> median)':  labs_missing_test > labs_median,
    'low vitals-missingness (<= median)': vitals_missing_test <= vitals_median,
    'high vitals-missingness (> median)': vitals_missing_test > vitals_median,
}

group_results = {}
print(f"\n{'Subgroup':40s} {'n':>5s} {'coverage':>9s} {'95% CI':>18s} {'mean set size':>14s}")
for name, mask in groups.items():
    n_g = mask.sum()
    if n_g == 0:
        continue
    cov_g = covered[mask].mean()
    lo, hi = wilson_ci(covered[mask].sum(), n_g)
    size_g = set_sizes[mask].mean()
    group_results[name] = {'n': int(n_g), 'coverage': float(cov_g), 'ci': (lo, hi), 'mean_size': float(size_g)}
    print(f"{name:40s} {n_g:5d} {cov_g:9.4f} [{lo:.3f}, {hi:.3f}] {size_g:14.4f}")

# Flag any subgroup whose CI excludes the nominal target as a conditional coverage gap (H1 test)
print(f"\nNominal target coverage: {1 - ALPHA:.4f}")
gaps_found = []
for name, res in group_results.items():
    lo, hi = res['ci']
    if not (lo <= (1 - ALPHA) <= hi):
        gaps_found.append(name)
        print(f"  CONDITIONAL COVERAGE GAP: '{name}' CI [{lo:.3f}, {hi:.3f}] excludes nominal "
              f"target {1 - ALPHA:.3f} -- supports H1 for this subgroup.")
if not gaps_found:
    print("  No subgroup CI excludes the nominal target at this sample size -- H1's conditional "
          "coverage gap is not detected here (may still exist but be underpowered to detect "
          "with this test-set size; consider bootstrapping or a larger cohort).")

# [5b/6] Diagnose the source of a gap: case-mix confound (subgroups differ in true
# mortality prevalence) vs. confidence confound (model predictions are systematically
# sharper/more hedged in one subgroup, e.g. due to more imputed inputs).
if gaps_found:
    print("\n[5b/6] Diagnosing the source of the conditional coverage gap(s)...")
    for name, mask in groups.items():
        if name not in group_results:
            continue
        prevalence_g = y_test_arr[mask].mean()
        p_mean_g = p_test[mask].mean()
        p_std_g  = p_test[mask].std()
        print(f"  {name:40s} mortality_rate={prevalence_g:.3f}  "
              f"pred_mean={p_mean_g:.3f}  pred_std={p_std_g:.3f}")
    print("  If mortality_rate is similar across the low/high split of a modality but "
          "pred_std differs a lot, the gap is more likely a confidence-sharpness artifact "
          "(mechanism b) than a genuine missingness-driven prevalence confound (mechanism a). "
          "Both still matter clinically, but they call for different fixes: (a) suggests "
          "re-weighting or stratified training; (b) is directly correctable with Mondrian "
          "(group-conditional) conformal calibration below.")


# [5c/6] Mondrian (group-conditional) conformal: calibrate a SEPARATE qhat within each
# missingness subgroup instead of one pooled qhat. This targets conditional coverage
# directly, at the cost of a smaller effective calibration sample per group.
print("\n[5c/6] Mondrian (group-conditional) conformal calibration...")

vitals_missing_calib = vitals_missing_frac[calib_idx]
labs_missing_calib    = labs_missing_frac[calib_idx]

mondrian_specs = {
    'labs_missingness':   (labs_missing_calib,   labs_missing_test,   labs_median),
    'vitals_missingness': (vitals_missing_calib, vitals_missing_test, vitals_median),
}

mondrian_results = {}
for spec_name, (calib_frac, test_frac, median_cut) in mondrian_specs.items():
    print(f"\n  Grouping by: {spec_name} (median split at {median_cut:.3f})")
    covered_mondrian = np.full(len(y_test_arr), np.nan)
    set_sizes_mondrian = np.full(len(y_test_arr), np.nan)
    for label, calib_mask, test_mask in [
        ('low',  calib_frac <= median_cut, test_frac <= median_cut),
        ('high', calib_frac >  median_cut, test_frac >  median_cut),
    ]:
        n_calib_g = calib_mask.sum()
        n_test_g  = test_mask.sum()
        if n_calib_g < 30 or n_test_g == 0:
            print(f"    {label:5s}: skipped (insufficient calib sample, n={n_calib_g})")
            continue
        qhat_g, pred_sets_g = conformal_binary(
            p_calib[calib_mask], y_calib_arr[calib_mask], p_test[test_mask], ALPHA)
        cov_g = pred_sets_g[np.arange(n_test_g), y_test_arr[test_mask].astype(int)]
        covered_mondrian[test_mask] = cov_g
        sizes_g = pred_sets_g.sum(axis=1)
        set_sizes_mondrian[test_mask] = sizes_g
        lo, hi = wilson_ci(cov_g.sum(), n_test_g)
        gap = not (lo <= (1 - ALPHA) <= hi)
        print(f"    {label:5s} (n_calib={n_calib_g}, n_test={n_test_g}): qhat={qhat_g:.4f}  "
              f"coverage={cov_g.mean():.4f}  95% CI=[{lo:.3f},{hi:.3f}]  mean_size={sizes_g.mean():.3f}  "
              f"{'STILL A GAP' if gap else 'fixed'}")
        if (sizes_g == 0).any():
            print(f"      NOTE: {(sizes_g==0).mean():.2%} of this group received an EMPTY "
                  f"prediction set (qhat={qhat_g:.4f} is tight enough that neither class clears "
                  f"the threshold for some patients) -- this is a distinct, more serious failure "
                  f"mode than a both-classes set: it means the coverage guarantee itself can be "
                  f"violated for those patients, not just that the model abstained.")
        else:
            print(f"      empty={(sizes_g==0).mean():.2%}  singleton={(sizes_g==1).mean():.2%}  "
                  f"both={(sizes_g==2).mean():.2%}")
    valid = ~np.isnan(covered_mondrian)
    if valid.any():
        print(f"    Overall coverage after Mondrian correction: {covered_mondrian[valid].mean():.4f}")
    mondrian_results[spec_name] = {
        'covered': covered_mondrian, 'set_sizes': set_sizes_mondrian
    }


# [5d/6] Robustness check: resample the calib/test split (patient-grouped, no
# retraining) to test whether the gap and Mondrian's fix hold beyond one split.
N_BOOTSTRAP = 100
print(f"\n[5d/6] Robustness check: resampling calib/test split {N_BOOTSTRAP} times "
      f"(patient-grouped, no retraining)...")

pool_idx = np.concatenate([calib_idx, test_idx])
p_pool = np.concatenate([p_calib, p_test])
y_pool = y[pool_idx]
subj_pool = subject_id[pool_idx]
vitals_missing_pool = vitals_missing_frac[pool_idx]
labs_missing_pool = labs_missing_frac[pool_idx]

calib_frac_of_pool = len(calib_idx) / len(pool_idx)

from sklearn.model_selection import GroupShuffleSplit

def bootstrap_once(seed):
    unique_subj = np.unique(subj_pool)
    gss = GroupShuffleSplit(n_splits=1, test_size=1 - calib_frac_of_pool, random_state=seed)
    calib_subj_pos, test_subj_pos = next(gss.split(unique_subj, groups=unique_subj))
    calib_subj_b = set(unique_subj[calib_subj_pos])
    test_subj_b  = set(unique_subj[test_subj_pos])
    calib_mask_b = np.isin(subj_pool, list(calib_subj_b))
    test_mask_b  = np.isin(subj_pool, list(test_subj_b))
    return calib_mask_b, test_mask_b

pooled_gap_flags = {'labs_missingness_high': [], 'vitals_missingness_high': []}
mondrian_gap_flags = {'labs_missingness_high': [], 'vitals_missingness_high': []}

for b in range(N_BOOTSTRAP):
    calib_mask_b, test_mask_b = bootstrap_once(seed=1000 + b)
    p_c, y_c = p_pool[calib_mask_b], y_pool[calib_mask_b]
    p_t, y_t = p_pool[test_mask_b], y_pool[test_mask_b]
    labs_c, labs_t = labs_missing_pool[calib_mask_b], labs_missing_pool[test_mask_b]
    vit_c, vit_t   = vitals_missing_pool[calib_mask_b], vitals_missing_pool[test_mask_b]

    qhat_b, pred_sets_b = conformal_binary(p_c, y_c, p_t, ALPHA)
    covered_b = pred_sets_b[np.arange(len(y_t)), y_t.astype(int)]

    for spec_name, (missing_c, missing_t) in [
        ('labs_missingness_high',   (labs_c, labs_t)),
        ('vitals_missingness_high', (vit_c, vit_t)),
    ]:
        median_b = np.median(missing_t)
        high_mask_t = missing_t > median_b
        high_mask_c = missing_c > median_b
        if high_mask_t.sum() < 10 or high_mask_c.sum() < 30:
            continue

        # pooled-qhat coverage on the high-missingness slice
        cov_high_pooled = covered_b[high_mask_t]
        lo, hi = wilson_ci(cov_high_pooled.sum(), len(cov_high_pooled))
        pooled_gap_flags[spec_name].append(not (lo <= (1 - ALPHA) <= hi))

        # Mondrian: recalibrate qhat using only the high-missingness calib slice
        qhat_high, pred_sets_high = conformal_binary(
            p_c[high_mask_c], y_c[high_mask_c], p_t[high_mask_t], ALPHA)
        cov_high_mondrian = pred_sets_high[np.arange(high_mask_t.sum()), y_t[high_mask_t].astype(int)]
        lo2, hi2 = wilson_ci(cov_high_mondrian.sum(), len(cov_high_mondrian))
        mondrian_gap_flags[spec_name].append(not (lo2 <= (1 - ALPHA) <= hi2))

for spec_name in pooled_gap_flags:
    n_valid = len(pooled_gap_flags[spec_name])
    if n_valid == 0:
        print(f"  {spec_name}: no valid bootstrap iterations (subgroup too small)")
        continue
    pooled_rate = np.mean(pooled_gap_flags[spec_name])
    mondrian_rate = np.mean(mondrian_gap_flags[spec_name])
    print(f"  {spec_name} (n={n_valid} valid resamples):")
    print(f"    Gap detected under POOLED calibration:   {pooled_rate:.1%} of resamples")
    print(f"    Gap detected under MONDRIAN calibration: {mondrian_rate:.1%} of resamples")
    if pooled_rate >= 0.7 and mondrian_rate <= 0.3:
        print(f"    -> Stable finding: pooled calibration reliably undercovers/overcovers this "
              f"subgroup, and Mondrian reliably fixes it.")
    elif pooled_rate < 0.5:
        print(f"    -> The original single-split gap may not be robust; it appeared in fewer "
              f"than half of resamples here.")
    else:
        print(f"    -> Mixed evidence; treat the single-split result cautiously and report "
              f"this resampling rate alongside it.")


# [5e/6] APS (Adaptive Prediction Sets, Romano et al. 2020) as a second conformal
# method. Uses the randomized score -- the deterministic variant is degenerate for
# binary classification (always returns the full {0,1} set).
print("\n[5e/6] APS (Adaptive Prediction Sets) as a second conformal method...")

aps_rng = np.random.RandomState(123)

def aps_scores_randomized(p, y, u):
    p = np.asarray(p); y = np.asarray(y); u = np.asarray(u)
    score_y1 = np.where(p >= 0.5, u * p, (1 - p) + u * p)
    score_y0 = np.where(p < 0.5, u * (1 - p), p + u * (1 - p))
    return np.where(y == 1, score_y1, score_y0), score_y0, score_y1

def aps_binary(p_calib, y_calib, p_test, alpha, rng):
    n = len(y_calib)
    u_calib = rng.uniform(0, 1, n)
    scores, _, _ = aps_scores_randomized(p_calib, y_calib, u_calib)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    qhat_local = np.quantile(scores, q_level, method='higher')
    u_test = rng.uniform(0, 1, len(p_test))
    _, score_0, score_1 = aps_scores_randomized(p_test, np.zeros_like(p_test), u_test)
    include_0 = score_0 <= qhat_local
    include_1 = score_1 <= qhat_local
    return qhat_local, np.stack([include_0, include_1], axis=1)

qhat_aps, pred_sets_aps = aps_binary(p_calib, y_calib_arr, p_test, ALPHA, aps_rng)
set_sizes_aps = pred_sets_aps.sum(axis=1)
covered_aps = pred_sets_aps[np.arange(len(y_test_arr)), y_test_arr.astype(int)]
ci_lo_aps, ci_hi_aps = wilson_ci(covered_aps.sum(), len(covered_aps))

print(f"  APS qhat: {qhat_aps:.4f}")
print(f"  APS marginal coverage: {covered_aps.mean():.4f}  95% CI=[{ci_lo_aps:.3f},{ci_hi_aps:.3f}]  "
      f"(target: {1-ALPHA:.3f})")
print(f"  APS mean set size: {set_sizes_aps.mean():.4f}  "
      f"(LAC mean set size for comparison: {set_sizes.mean():.4f})")
print(f"  APS set size distribution -> empty:{(set_sizes_aps==0).mean():.2%}  "
      f"singleton:{(set_sizes_aps==1).mean():.2%}  both-classes:{(set_sizes_aps==2).mean():.2%}")
print("  Note: APS is designed to be more adaptive to per-example uncertainty than LAC, "
      "typically at the cost of larger average set size -- the comparison above is the "
      "efficiency/adaptivity trade-off between the two methods, not a correctness check "
      "(both should hit marginal coverage; LAC is expected to have the smaller mean size).")


# [5f/6] Calibration diagrams: reliability diagrams + ECE/MCE for raw vs. isotonic-
# calibrated probabilities.
print("\n[5f/6] Calibration diagrams (reliability + ECE/MCE)...")

def ece_mce(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    ece, mce = 0.0, 0.0
    bin_centers, bin_acc, bin_conf, bin_counts = [], [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = mask.sum()
        if n_b == 0:
            continue
        acc_b = labels[mask].mean()
        conf_b = probs[mask].mean()
        gap = abs(acc_b - conf_b)
        ece += (n_b / len(probs)) * gap
        mce = max(mce, gap)
        bin_centers.append((bins[b] + bins[b+1]) / 2)
        bin_acc.append(acc_b)
        bin_conf.append(conf_b)
        bin_counts.append(n_b)
    return ece, mce, np.array(bin_centers), np.array(bin_acc), np.array(bin_conf), np.array(bin_counts)

raw_test = model.predict(X_scaled[test_idx], verbose=0).flatten()
ece_raw, mce_raw, bc_raw, ba_raw, bconf_raw, _ = ece_mce(raw_test, y_test_arr)
ece_cal, mce_cal, bc_cal, ba_cal, bconf_cal, _ = ece_mce(p_test, y_test_arr)

print(f"  Raw model probabilities:        ECE={ece_raw:.4f}  MCE={mce_raw:.4f}")
print(f"  Isotonic-calibrated probabilities: ECE={ece_cal:.4f}  MCE={mce_cal:.4f}")
print(f"  ({'Calibration improved' if ece_cal < ece_raw else 'Calibration did not improve'} "
      f"ECE on the test set.)")

fig_cal, axes_cal = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, bc, ba, ece_v, mce_v, title in [
    (axes_cal[0], bc_raw, ba_raw, ece_raw, mce_raw, 'Raw model probabilities'),
    (axes_cal[1], bc_cal, ba_cal, ece_cal, mce_cal, 'Isotonic-calibrated probabilities'),
]:
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect calibration')
    if len(bc) > 0:
        ax.plot(bc, ba, 'o-', color='#457b9d')
    ax.set_xlabel('Mean predicted probability (bin)')
    ax.set_ylabel('Observed mortality rate (bin)')
    ax.set_title(f'{title}\nECE={ece_v:.3f}  MCE={mce_v:.3f}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/kaggle/working/results/reliability_diagrams.png', dpi=100, bbox_inches='tight')
plt.close()
print("  Saved: /kaggle/working/results/reliability_diagrams.png")


# [5g/6] Temporal analysis: coverage stability across ICU stay length (short vs. long).
print("\n[5g/6] Temporal analysis: coverage by ICU stay length...")

los_path = '/kaggle/working/los_hours.npy'
if os.path.exists(los_path):
    los_hours = np.load(los_path)
    los_test = los_hours[test_idx]
    los_calib = los_hours[calib_idx]
    los_median = np.median(los_test)

    stay_groups = {
        f'short stay (<= {los_median:.0f}h)': los_test <= los_median,
        f'long stay (> {los_median:.0f}h)':   los_test > los_median,
    }
    print(f"\n{'Subgroup':30s} {'n':>5s} {'coverage':>9s} {'95% CI':>18s} {'mean set size':>14s}")
    stay_group_results = {}
    stay_gaps_found = []
    for name, mask in stay_groups.items():
        n_g = mask.sum()
        if n_g == 0:
            continue
        cov_g = covered[mask].mean()
        lo, hi = wilson_ci(covered[mask].sum(), n_g)
        size_g = set_sizes[mask].mean()
        gap = not (lo <= (1 - ALPHA) <= hi)
        stay_group_results[name] = {'mask': mask, 'coverage': cov_g, 'ci': (lo, hi), 'n': int(n_g), 'gap': gap}
        if gap:
            stay_gaps_found.append(name)
        print(f"{name:30s} {n_g:5d} {cov_g:9.4f} [{lo:.3f}, {hi:.3f}] {size_g:14.4f} "
              f"{'GAP' if gap else 'ok'}")

    if stay_gaps_found:
        print(f"\n  RESULT: coverage is NOT stable across stay length -- "
              f"{len(stay_gaps_found)}/{len(stay_group_results)} subgroup(s) fall outside the "
              f"nominal {1-ALPHA:.0%} target: {', '.join(stay_gaps_found)}. This directly "
              f"answers the temporal-stability question the AR(1) simulation was meant to "
              f"probe: conformal coverage measurably degrades/inflates depending on ICU stay "
              f"length, which is itself a real, discoverable temporal effect in this cohort.")
        print("  Diagnosing the mechanism (mirrors the missingness diagnostic in [5b/6])...")
        for name, res in stay_group_results.items():
            prevalence_g = y_test_arr[res['mask']].mean()
            p_mean_g = p_test[res['mask']].mean()
            p_std_g = p_test[res['mask']].std()
            print(f"    {name:30s} mortality_rate={prevalence_g:.3f}  "
                  f"pred_mean={p_mean_g:.3f}  pred_std={p_std_g:.3f}")
    else:
        print(f"\n  RESULT: coverage is stable across stay length -- both subgroup CIs contain "
              f"the nominal {1-ALPHA:.0%} target. The conformal guarantee does not measurably "
              f"degrade for longer stays relative to shorter ones at this sample size, which "
              f"is the practical question the proposed AR(1) simulation was aiming to answer.")
else:
    print("  SKIPPED: los_hours.npy not found (requires the updated Stage 1 script to have "
          "been re-run). Re-run Stage 1 to generate it, then re-run Stage 3.")
    stay_group_results = {}


# [5g2/6] Correlation check: does stay length share an underlying confound with
# missingness and mortality (i.e. clinical acuity), rather than being independent?
if os.path.exists(los_path):
    print("\n[5g2/6] Checking whether stay length and missingness share a common confound...")

    pairs = [
        ('stay length vs labs missingness', los_test, labs_missing_test),
        ('stay length vs vitals missingness', los_test, vitals_missing_test),
        ('stay length vs mortality', los_test, y_test_arr.astype(float)),
        ('labs missingness vs mortality', labs_missing_test, y_test_arr.astype(float)),
        ('vitals missingness vs mortality', vitals_missing_test, y_test_arr.astype(float)),
    ]
    print(f"\n{'Pair':38s} {'Pearson r':>10s} {'p-value':>10s} {'Spearman r':>11s} {'p-value':>10s}")
    for name, a, b in pairs:
        r_p, p_p = pearsonr(a, b)
        r_s, p_s = spearmanr(a, b)
        print(f"{name:38s} {r_p:10.3f} {p_p:10.2e} {r_s:11.3f} {p_s:10.2e}")

    print("\n  Interpretation: a negative stay-length/missingness correlation (longer stays have "
          "LESS missing data) combined with same-direction correlations to mortality for both "
          "stay length and missingness would indicate a single shared driver -- most plausibly "
          "clinical acuity, since sicker patients are monitored more closely (lower missingness) "
          "and remain in the ICU longer (longer stay), and also die more often. If that pattern "
          "holds, report stay-length and missingness as two manifestations of one confound in "
          "Discussion, rather than as independent findings -- it is a more precise and more "
          "defensible claim than treating them as separate effects.")
    composite_low_mask = composite_high_mask = None
else:
    print("\n[5g2/6] SKIPPED (requires los_hours.npy, see [5g/6] above).")
    composite_low_mask = composite_high_mask = None


# [5g3/6] Composite acuity Mondrian correction: combine stay length and missingness
# into a single severity proxy and test Mondrian calibration on that composite.
# Z-scores fit on the calibration set only, to avoid leakage.
if os.path.exists(los_path):
    print("\n[5g3/6] Composite acuity grouping (stay length + missingness combined)...")

    def _z(x, mu, sd):
        return (x - mu) / sd if sd > 0 else np.zeros_like(x)

    mu_los, sd_los = los_calib.mean(), los_calib.std()
    mu_labs, sd_labs = labs_missing_calib.mean(), labs_missing_calib.std()
    mu_vit, sd_vit = vitals_missing_calib.mean(), vitals_missing_calib.std()

    # Higher composite = more acute: longer stay AND less missing data (more monitoring)
    acuity_calib = (_z(los_calib, mu_los, sd_los)
                     - _z(labs_missing_calib, mu_labs, sd_labs)
                     - _z(vitals_missing_calib, mu_vit, sd_vit))
    acuity_test = (_z(los_test, mu_los, sd_los)
                    - _z(labs_missing_test, mu_labs, sd_labs)
                    - _z(vitals_missing_test, mu_vit, sd_vit))
    acuity_median = np.median(acuity_test)

    composite_low_mask = acuity_test <= acuity_median    # lower acuity
    composite_high_mask = acuity_test > acuity_median    # higher acuity
    composite_low_calib_mask = acuity_calib <= acuity_median
    composite_high_calib_mask = acuity_calib > acuity_median

    print(f"  Composite groups: low-acuity n={composite_low_mask.sum()}, "
          f"high-acuity n={composite_high_mask.sum()}")

    print(f"\n  BEFORE (pooled qhat={qhat:.4f}, same as marginal [4/6]):")
    for name, mask in [('low acuity', composite_low_mask), ('high acuity', composite_high_mask)]:
        cov_g = covered[mask].mean()
        lo, hi = wilson_ci(covered[mask].sum(), mask.sum())
        gap = not (lo <= (1 - ALPHA) <= hi)
        print(f"    {name:15s} n={mask.sum():5d}  coverage={cov_g:.4f}  CI=[{lo:.3f},{hi:.3f}]  "
              f"{'GAP' if gap else 'ok'}")

    print(f"\n  AFTER (Mondrian on composite acuity):")
    composite_mondrian_covered = np.full(len(y_test_arr), np.nan)
    composite_mondrian_set_sizes = np.full(len(y_test_arr), np.nan)
    for name, cmask, tmask in [('low acuity', composite_low_calib_mask, composite_low_mask),
                                ('high acuity', composite_high_calib_mask, composite_high_mask)]:
        n_c, n_t = cmask.sum(), tmask.sum()
        if n_c < 30 or n_t == 0:
            print(f"    {name:15s} skipped (insufficient calib sample, n={n_c})")
            continue
        qhat_g, sets_g = conformal_binary(p_calib[cmask], y_calib_arr[cmask], p_test[tmask], ALPHA)
        cov_g = sets_g[np.arange(n_t), y_test_arr[tmask].astype(int)]
        composite_mondrian_covered[tmask] = cov_g
        sizes_g = sets_g.sum(axis=1)
        composite_mondrian_set_sizes[tmask] = sizes_g
        lo, hi = wilson_ci(cov_g.sum(), n_t)
        gap = not (lo <= (1 - ALPHA) <= hi)
        print(f"    {name:15s} n_calib={n_c:5d} n_test={n_t:5d}  qhat={qhat_g:.4f}  "
              f"coverage={cov_g.mean():.4f}  CI=[{lo:.3f},{hi:.3f}]  mean_size={sizes_g.mean():.3f}  "
              f"{'STILL A GAP' if gap else 'fixed'}")
        if (sizes_g == 0).any():
            print(f"      NOTE: {(sizes_g==0).mean():.2%} of this group received an EMPTY "
                  f"prediction set (qhat={qhat_g:.4f} tight enough that neither class clears "
                  f"the threshold for some patients) -- a more serious failure mode than a "
                  f"both-classes set, since the coverage guarantee itself can fail for these "
                  f"patients rather than the model simply abstaining.")
        else:
            print(f"      empty={(sizes_g==0).mean():.2%}  singleton={(sizes_g==1).mean():.2%}  "
                  f"both={(sizes_g==2).mean():.2%}")
    valid = ~np.isnan(composite_mondrian_covered)
    if valid.any():
        print(f"  Overall coverage after composite Mondrian correction: "
              f"{composite_mondrian_covered[valid].mean():.4f}")
    print("  Compare the residual gap sizes here against the single-variable Mondrian results "
          "in [5c/6] -- if the composite grouping leaves a SMALLER residual gap (tighter CIs "
          "around the 90% target) than missingness or stay-length alone, that supports treating "
          "acuity as the more fundamental grouping variable for this deployment context.")
else:
    print("\n[5g3/6] SKIPPED (requires los_hours.npy, see [5g/6] above).")


# [5j/6] Demographic fairness: conditional coverage by age and sex.
print("\n[5j/6] Demographic fairness: coverage by age and sex...")

age_path = '/kaggle/working/age.npy'
gender_path = '/kaggle/working/gender.npy'
fairness_group_results = {}

if os.path.exists(age_path) and os.path.exists(gender_path):
    age = np.load(age_path)
    gender = np.load(gender_path, allow_pickle=True)
    age_test = age[test_idx]
    gender_test = gender[test_idx]
    age_median = np.median(age_test)

    fairness_groups = {
        f'younger (<= {age_median:.0f}y)': age_test <= age_median,
        f'older (> {age_median:.0f}y)':    age_test > age_median,
        'male':   gender_test == 'M',
        'female': gender_test == 'F',
    }

    print(f"\n{'Subgroup':22s} {'n':>5s} {'coverage':>9s} {'95% CI':>18s} {'mean set size':>14s}")
    fairness_gaps_found = []
    for name, mask in fairness_groups.items():
        n_g = mask.sum()
        if n_g == 0:
            print(f"{name:22s}     0  (no patients in this group, skipped)")
            continue
        cov_g = covered[mask].mean()
        lo, hi = wilson_ci(covered[mask].sum(), n_g)
        size_g = set_sizes[mask].mean()
        gap = not (lo <= (1 - ALPHA) <= hi)
        fairness_group_results[name] = {'mask': mask, 'coverage': cov_g, 'ci': (lo, hi), 'n': int(n_g), 'gap': gap}
        if gap:
            fairness_gaps_found.append(name)
        print(f"{name:22s} {n_g:5d} {cov_g:9.4f} [{lo:.3f}, {hi:.3f}] {size_g:14.4f} "
              f"{'GAP' if gap else 'ok'}")
        print(f"    mortality_rate={y_test_arr[mask].mean():.3f}  "
              f"pred_mean={p_test[mask].mean():.3f}  pred_std={p_test[mask].std():.3f}")

    if fairness_gaps_found:
        print(f"\n  RESULT: conditional coverage is NOT uniform across demographic subgroups -- "
              f"{len(fairness_gaps_found)} subgroup(s) fall outside the nominal {1-ALPHA:.0%} "
              f"target: {', '.join(fairness_gaps_found)}. This is a fairness-relevant finding: "
              f"if a subgroup systematically undercovers, the conformal guarantee is being "
              f"delivered unevenly across a protected characteristic, not just across clinical "
              f"subgroups. Compare against the mortality_rate/pred_std diagnostics above to check "
              f"whether this traces to the same case-mix/acuity mechanism as the missingness and "
              f"stay-length gaps, or is a distinct effect.")
    else:
        print(f"\n  RESULT: coverage is stable across age and sex subgroups at this sample size -- "
              f"no evidence of a demographic fairness gap in conditional coverage (though absence "
              f"of evidence is not evidence of absence at these subgroup sizes; see Limitations).")
else:
    print("  SKIPPED: age.npy/gender.npy not found (requires the updated Stage 1 script to have "
          "been re-run). Re-run Stage 1 to generate them, then re-run Stage 3.")


# [5k/6] Class-conditional coverage (died vs. survived).
print("\n[5k/6] Class-conditional coverage (died vs. survived)...")

for cls, label in [(0, 'survived'), (1, 'died')]:
    mask = y_test_arr == cls
    n_c = mask.sum()
    if n_c == 0:
        continue
    cov_c = covered[mask].mean()
    lo, hi = wilson_ci(covered[mask].sum(), n_c)
    gap = not (lo <= (1 - ALPHA) <= hi)
    print(f"  {label:10s} (y={cls}): n={n_c:5d}  coverage={cov_c:.4f}  95% CI=[{lo:.3f},{hi:.3f}]  "
          f"{'GAP' if gap else 'ok'}")
print("  Compare against Gouripeddi et al. (2026): 99.7% coverage (death class) vs. "
      "81.1% (survived class) in an LLM-on-notes conformal pipeline.")


# [5l/6] Conformal prediction using raw vs. calibrated probabilities: does
# calibration improve conformal efficiency (set size), not just point-probability
# calibration quality? Reuses raw_test from [5f/6], no retraining needed.
print("\n[5l/6] Conformal prediction using RAW (uncalibrated) vs. calibrated probabilities...")

raw_calib = model.predict(X_scaled[calib_idx], verbose=0).flatten()
qhat_raw, pred_sets_raw = conformal_binary(raw_calib, y_calib_arr, raw_test, ALPHA)
covered_raw = pred_sets_raw[np.arange(len(y_test_arr)), y_test_arr.astype(int)]
set_sizes_raw = pred_sets_raw.sum(axis=1)
lo_raw, hi_raw = wilson_ci(covered_raw.sum(), len(covered_raw))

print(f"  Raw probabilities:        qhat={qhat_raw:.4f}  coverage={covered_raw.mean():.4f}  "
      f"95% CI=[{lo_raw:.3f},{hi_raw:.3f}]  mean_size={set_sizes_raw.mean():.4f}")
print(f"  Calibrated probabilities: qhat={qhat:.4f}  coverage={covered.mean():.4f}  "
      f"mean_size={set_sizes.mean():.4f}  (from [4/6] above)")
print("  Both rows should hit ~90% coverage regardless of calibration quality -- the "
      "conformal guarantee doesn't depend on it. What calibration buys is efficiency: "
      "a meaningfully smaller mean_size for calibrated vs. raw is direct evidence "
      "isotonic calibration produces tighter, more clinically useful prediction sets.")


# [5h/6] Formal significance testing for subgroup coverage differences: Wilcoxon
# rank-sum plus a permutation test as a model-free cross-check.
print("\n[5h/6] Formal significance testing for subgroup coverage differences...")

def permutation_test_coverage_diff(covered_a, covered_b, n_perm=5000, rng=None):
    rng = rng or np.random.RandomState(0)
    observed_diff = covered_a.mean() - covered_b.mean()
    pooled = np.concatenate([covered_a, covered_b])
    n_a = len(covered_a)
    diffs = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(pooled)
        diffs[i] = perm[:n_a].mean() - perm[n_a:].mean()
    p_value = (np.abs(diffs) >= np.abs(observed_diff)).mean()
    return observed_diff, p_value

sig_test_pairs = []
if 'low labs-missingness (<= median)' in groups and 'high labs-missingness (> median)' in groups:
    sig_test_pairs.append(('labs missingness: low vs high',
                            covered[groups['low labs-missingness (<= median)']],
                            covered[groups['high labs-missingness (> median)']]))
if 'low vitals-missingness (<= median)' in groups and 'high vitals-missingness (> median)' in groups:
    sig_test_pairs.append(('vitals missingness: low vs high',
                            covered[groups['low vitals-missingness (<= median)']],
                            covered[groups['high vitals-missingness (> median)']]))
if len(stay_group_results) == 2:
    names = list(stay_group_results.keys())
    sig_test_pairs.append((f'stay length: {names[0]} vs {names[1]}',
                            covered[stay_group_results[names[0]]['mask']],
                            covered[stay_group_results[names[1]]['mask']]))
if composite_low_mask is not None and composite_high_mask is not None:
    sig_test_pairs.append(('composite acuity: low vs high',
                            covered[composite_low_mask],
                            covered[composite_high_mask]))
if fairness_group_results:
    age_names = [n for n in fairness_group_results if 'younger' in n or 'older' in n]
    if len(age_names) == 2:
        sig_test_pairs.append((f'age: {age_names[0]} vs {age_names[1]}',
                                covered[fairness_group_results[age_names[0]]['mask']],
                                covered[fairness_group_results[age_names[1]]['mask']]))
    if 'male' in fairness_group_results and 'female' in fairness_group_results:
        sig_test_pairs.append(('sex: male vs female',
                                covered[fairness_group_results['male']['mask']],
                                covered[fairness_group_results['female']['mask']]))

perm_rng = np.random.RandomState(7)
print(f"\n{'Comparison':40s} {'coverage diff':>14s} {'Wilcoxon p':>12s} {'Permutation p':>14s}")
for name, cov_a, cov_b in sig_test_pairs:
    try:
        _, p_wilcoxon = mannwhitneyu(cov_a.astype(int), cov_b.astype(int), alternative='two-sided')
    except ValueError:
        p_wilcoxon = np.nan  # identical distributions in both groups
    diff, p_perm = permutation_test_coverage_diff(cov_a, cov_b, n_perm=5000, rng=perm_rng)
    sig_marker = '*' if (not np.isnan(p_wilcoxon) and p_wilcoxon < 0.05) or p_perm < 0.05 else ''
    print(f"{name:40s} {diff:14.4f} {p_wilcoxon:12.4f} {p_perm:14.4f} {sig_marker}")
print("  (* = significant at p<0.05 under at least one test)")


# [5i/6] Consolidated prediction-set-size distributions with clinical utility framing.
print("\n[5i/6] Consolidated set-size distributions and clinical utility...")

def report_set_size(name, sizes):
    sizes = sizes[~np.isnan(sizes)]
    n = len(sizes)
    if n == 0:
        print(f"  {name:35s} (no valid data)")
        return
    print(f"  {name:35s} n={n:5d}  empty={np.mean(sizes==0):.2%}  "
          f"singleton={np.mean(sizes==1):.2%}  both={np.mean(sizes==2):.2%}  "
          f"mean={sizes.mean():.3f}")

print("  -- Pooled qhat (before any group-conditional correction) --")
report_set_size('Marginal (LAC, all test patients)', set_sizes)
report_set_size('Marginal (APS, all test patients)', set_sizes_aps)
for name, mask in groups.items():
    if mask.sum() > 0:
        report_set_size(name, set_sizes[mask])
for name, res in stay_group_results.items():
    report_set_size(name, set_sizes[res['mask']])

print("\n  -- Mondrian-corrected (after group-conditional calibration) --")
print("  (compare against the pooled-qhat rows above for the same subgroup: Mondrian changes")
print("   the full set-size distribution, not just the coverage number -- including, sometimes,")
print("   introducing empty sets that don't appear anywhere under the pooled qhat)")
if 'labs_missingness' in mondrian_results:
    report_set_size('Mondrian: labs missingness (both groups combined)',
                     mondrian_results['labs_missingness']['set_sizes'])
if 'vitals_missingness' in mondrian_results:
    report_set_size('Mondrian: vitals missingness (both groups combined)',
                     mondrian_results['vitals_missingness']['set_sizes'])
if 'composite_mondrian_set_sizes' in globals():
    report_set_size('Mondrian: composite acuity (both groups combined)',
                     composite_mondrian_set_sizes)

singleton_rate = (set_sizes == 1).mean()
both_rate = (set_sizes == 2).mean()
empty_rate = (set_sizes == 0).mean()
print(f"\n  Clinical utility framing (pooled LAC, marginal): {singleton_rate:.1%} of test patients "
      f"receive a singleton prediction set -- i.e. the model commits to a single risk class at "
      f"the target coverage level, which is directly actionable. {both_rate:.1%} receive a "
      f"both-classes set, a principled abstention signal rather than a failure mode: it flags "
      f"patients where committing to a single class would not carry the guaranteed validity the "
      f"method promises, and are natural candidates for escalation to clinician review. "
      f"{f'A further {empty_rate:.1%} received an EMPTY set, ' if empty_rate > 0 else ''}"
      f"Note that group-conditional (Mondrian) calibration can introduce empty sets even where "
      f"the pooled/marginal analysis has none (see composite acuity, low-acuity group above) -- "
      f"an empty set is a materially different, more serious outcome than a both-classes "
      f"abstention, since it means the coverage guarantee itself can be violated for that "
      f"patient rather than the model declining to commit. This trade-off (tighter subgroup "
      f"coverage at the cost of occasional empty sets) is worth stating explicitly whenever "
      f"Mondrian correction is proposed for deployment, not just its effect on coverage alone.")


# [6/6] Save results and plots
print("\n[6/6] Saving results...")
os.makedirs('/kaggle/working/results', exist_ok=True)
os.makedirs('/kaggle/working/models', exist_ok=True)

np.savez('/kaggle/working/results/conformal_results.npz',
         qhat=qhat, pred_sets=pred_sets, set_sizes=set_sizes, covered=covered,
         p_test=p_test, y_test=y_test_arr, test_idx=test_idx,
         vitals_missing_test=vitals_missing_test, labs_missing_test=labs_missing_test,
         alpha=ALPHA,
         mondrian_labs_covered=mondrian_results['labs_missingness']['covered'],
         mondrian_vitals_covered=mondrian_results['vitals_missingness']['covered'],
         bootstrap_pooled_gap_rate_labs=np.mean(pooled_gap_flags['labs_missingness_high']) if pooled_gap_flags['labs_missingness_high'] else np.nan,
         bootstrap_mondrian_gap_rate_labs=np.mean(mondrian_gap_flags['labs_missingness_high']) if mondrian_gap_flags['labs_missingness_high'] else np.nan,
         bootstrap_pooled_gap_rate_vitals=np.mean(pooled_gap_flags['vitals_missingness_high']) if pooled_gap_flags['vitals_missingness_high'] else np.nan,
         bootstrap_mondrian_gap_rate_vitals=np.mean(mondrian_gap_flags['vitals_missingness_high']) if mondrian_gap_flags['vitals_missingness_high'] else np.nan,
         qhat_aps=qhat_aps, pred_sets_aps=pred_sets_aps, set_sizes_aps=set_sizes_aps, covered_aps=covered_aps,
         ece_raw=ece_raw, mce_raw=mce_raw, ece_calibrated=ece_cal, mce_calibrated=mce_cal,
         stay_length_median_hours=los_median if os.path.exists(los_path) else np.nan)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].bar(['empty', 'singleton', 'both classes'],
            [(set_sizes==0).mean(), (set_sizes==1).mean(), (set_sizes==2).mean()],
            color=['#999999', '#2a9d8f', '#e76f51'])
axes[0].set_title('Prediction set size distribution (test set)')
axes[0].set_ylabel('Fraction of patients')
axes[0].grid(alpha=0.3)

names = list(group_results.keys())
covs  = [group_results[n]['coverage'] for n in names]
los   = [group_results[n]['ci'][0] for n in names]
his   = [group_results[n]['ci'][1] for n in names]
errs  = [[c-l for c,l in zip(covs,los)], [h-c for c,h in zip(covs,his)]]
axes[1].barh(names, covs, xerr=errs, color='#457b9d', capsize=4)
axes[1].axvline(1 - ALPHA, color='red', linestyle='--', label=f'nominal target ({1-ALPHA:.0%})')
axes[1].set_xlabel('Coverage')
axes[1].set_title('Conditional coverage by missingness subgroup\n(95% Wilson CI)')
axes[1].legend(loc='lower right', fontsize=8)
axes[1].grid(alpha=0.3)

axes[2].scatter(labs_missing_test, set_sizes + np.random.uniform(-0.05, 0.05, len(set_sizes)),
                 alpha=0.4, s=15, c=covered, cmap='RdYlGn', vmin=0, vmax=1)
axes[2].set_xlabel('Labs missingness fraction (pre-imputation)')
axes[2].set_ylabel('Prediction set size (jittered)')
axes[2].set_title('Set size vs. labs missingness\n(color = covered)')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/results/conformal_coverage.png', dpi=100, bbox_inches='tight')
plt.close()

print("\nSaved:")
print("  /kaggle/working/results/conformal_results.npz")
print("  /kaggle/working/results/conformal_coverage.png")
print("STAGE 3 COMPLETE")
