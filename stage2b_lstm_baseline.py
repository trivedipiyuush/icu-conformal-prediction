import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, precision_score,
                              recall_score, brier_score_loss, precision_recall_curve)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import joblib
import os

np.random.seed(42)
tf.random.set_seed(42)

print("STAGE 2B: LSTM BASELINE (real MIMIC-IV data, same split as Transformer)")

ALPHA = 0.10

print("\n[1/7] Loading Stage 1 features and the corrected patient-grouped split...")
REQUIRED = ['/kaggle/working/X.npy', '/kaggle/working/y.npy', '/kaggle/working/subject_id.npy']
missing = [f for f in REQUIRED if not os.path.exists(f)]
if missing:
    raise FileNotFoundError(
        f"Missing required Stage 1 output(s): {missing}. Re-run Stage 1 in this session first.")

X = np.load('/kaggle/working/X.npy')
y = np.load('/kaggle/working/y.npy')
subject_id = np.load('/kaggle/working/subject_id.npy')
n_patients, n_hours, n_features = X.shape
print(f"X: {X.shape}  |  y: {y.shape}  |  mortality: {y.mean():.2%}")

corrected_split_path = '/kaggle/working/corrected_split_indices.npz'
split_indices_path = '/kaggle/working/split_indices.npz'

if os.path.exists(corrected_split_path):
    split = np.load(corrected_split_path)
    train_idx, calib_idx, test_idx = split['train_idx'], split['calib_idx'], split['test_idx']
    print(f"Loaded corrected split from Stage 3 (was_corrected={bool(split['was_corrected'])}): "
          f"train:{len(train_idx)} calib:{len(calib_idx)} test:{len(test_idx)}")
else:
    print("WARNING: /kaggle/working/corrected_split_indices.npz not found. Computing the "
          "patient-grouped calib/test correction directly, but using Stage 2's original "
          "train_idx (from split_indices.npz) so the training set matches the Transformer's "
          "regardless of run order.")
    if not os.path.exists(split_indices_path):
        raise FileNotFoundError(
            "Missing /kaggle/working/split_indices.npz. Run Stage 2 first, in this session.")
    train_idx = np.load(split_indices_path)['train_idx']

    unique_subjects = np.unique(subject_id)
    subj_to_label = {s: y[subject_id == s].max() for s in unique_subjects}
    labels_per_subject = np.array([subj_to_label[s] for s in unique_subjects])
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tc_pos, test_pos = next(gss1.split(unique_subjects, labels_per_subject, groups=unique_subjects))
    tc_subj, test_subj = unique_subjects[tc_pos], unique_subjects[test_pos]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr_pos, cal_pos = next(gss2.split(tc_subj, labels_per_subject[tc_pos], groups=tc_subj))
    calib_subj = tc_subj[cal_pos]
    calib_idx = np.where(np.isin(subject_id, calib_subj))[0]
    test_idx = np.where(np.isin(subject_id, test_subj))[0]
    print(f"Using: train:{len(train_idx)} (Stage 2 original) calib:{len(calib_idx)} test:{len(test_idx)} (patient-grouped)")

assert len(set(subject_id[calib_idx]) & set(subject_id[test_idx])) == 0, \
    "Calib/test patient overlap detected -- this split is not valid for a fair comparison."

print("\n[2/7] Normalizing per-feature (fit on train only)...")
scaler = StandardScaler()
X_train_2d = X[train_idx].reshape(-1, n_features)
scaler.fit(X_train_2d)
X_2d = X.reshape(-1, n_features)
X_scaled = scaler.transform(X_2d).reshape(n_patients, n_hours, n_features).astype('float32')

X_train, y_train = X_scaled[train_idx], y[train_idx]
X_calib, y_calib = X_scaled[calib_idx], y[calib_idx]
X_test, y_test = X_scaled[test_idx], y[test_idx]

pos_weight = (1 - y_train.mean()) / (y_train.mean() + 1e-6)
class_weight = {0: 1.0, 1: float(pos_weight)}
print(f"Class weight (positive): {class_weight[1]:.2f}")

print("\n[3/7] Building baseline LSTM...")

def build_baseline_lstm(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        # name='auc' avoids Keras storing the history key as 'AUC' (capitalized)
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    return model

model = build_baseline_lstm(input_shape=(n_hours, n_features))
print(f"Model parameters: {model.count_params():,}")

print("\n[4/7] Training (fixed 60-epoch budget, no early stopping -- best-AUC weights "
      "restored via checkpoint, so training length is no longer a source of "
      "run-to-run variation)...")
FIXED_EPOCHS = 60
checkpoint_path = '/kaggle/working/models/lstm_best_checkpoint.weights.h5'
os.makedirs('/kaggle/working/models', exist_ok=True)
checkpoint = callbacks.ModelCheckpoint(checkpoint_path, monitor='val_auc', mode='max',
                                       save_best_only=True, save_weights_only=True, verbose=0)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5,
                                        patience=5, min_lr=1e-6, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_calib, y_calib),
    epochs=FIXED_EPOCHS,
    batch_size=32,
    class_weight=class_weight,
    callbacks=[checkpoint, reduce_lr],
    verbose=0
)
model.load_weights(checkpoint_path)
best_epoch = int(np.argmax(history.history['val_auc'])) + 1
print(f"Trained for the full fixed budget: {FIXED_EPOCHS} epochs")
print(f"Best val_auc: {max(history.history['val_auc']):.4f} (epoch {best_epoch}/{FIXED_EPOCHS}, weights restored)")

print("\n[5/7] Evaluating and calibrating (same procedure as the Transformer)...")
y_calib_pred = model.predict(X_calib, verbose=0).flatten()
y_test_pred = model.predict(X_test, verbose=0).flatten()

iso_calibrator = IsotonicRegression(out_of_bounds='clip')
iso_calibrator.fit(y_calib_pred, y_calib)
platt_calibrator = LogisticRegression()
platt_calibrator.fit(y_calib_pred.reshape(-1, 1), y_calib)

y_calib_iso = iso_calibrator.predict(y_calib_pred)
y_test_iso = iso_calibrator.predict(y_test_pred)
y_calib_platt = platt_calibrator.predict_proba(y_calib_pred.reshape(-1, 1))[:, 1]
y_test_platt = platt_calibrator.predict_proba(y_test_pred.reshape(-1, 1))[:, 1]

brier_raw = brier_score_loss(y_calib, y_calib_pred)
brier_iso = brier_score_loss(y_calib, y_calib_iso)
brier_platt = brier_score_loss(y_calib, y_calib_platt)
best_calib = min({'raw': brier_raw, 'isotonic': brier_iso, 'platt': brier_platt},
                  key={'raw': brier_raw, 'isotonic': brier_iso, 'platt': brier_platt}.get)
print(f"Brier (calib) -> raw:{brier_raw:.4f} isotonic:{brier_iso:.4f} platt:{brier_platt:.4f}  "
      f"selected: {best_calib}")

if best_calib == 'isotonic':
    y_calib_final, y_test_final, calibrator_obj = y_calib_iso, y_test_iso, iso_calibrator
elif best_calib == 'platt':
    y_calib_final, y_test_final, calibrator_obj = y_calib_platt, y_test_platt, platt_calibrator
else:
    y_calib_final, y_test_final, calibrator_obj = y_calib_pred, y_test_pred, None

precisions, recalls, pr_thresholds = precision_recall_curve(y_calib, y_calib_final)
f1_scores = np.divide(2*precisions*recalls, precisions+recalls,
                       out=np.zeros_like(precisions), where=(precisions+recalls) != 0)
best_idx = np.argmax(f1_scores[:-1])
threshold = pr_thresholds[best_idx]

auc = roc_auc_score(y_test, y_test_pred)
f1 = f1_score(y_test, (y_test_final > threshold).astype(int), zero_division=0)
acc = accuracy_score(y_test, (y_test_final > threshold).astype(int))
prec = precision_score(y_test, (y_test_final > threshold).astype(int), zero_division=0)
rec = recall_score(y_test, (y_test_final > threshold).astype(int), zero_division=0)

print("\nLSTM BASELINE TEST RESULTS (real data, same split as Transformer)")
print(f"AUC:       {auc:.4f}")
print(f"F1:        {f1:.4f}")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"Threshold: {threshold:.4f} ({best_calib})")

print("\n[6/7] Applying the same LAC conformal wrapper as the Transformer...")

def conformal_binary(p_calib, y_calib, p_test, alpha):
    n = len(y_calib)
    scores = 1 - np.where(y_calib == 1, p_calib, 1 - p_calib)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    qhat = np.quantile(scores, q_level, method='higher')
    include_0 = p_test <= qhat
    include_1 = (1 - p_test) <= qhat
    return qhat, np.stack([include_0, include_1], axis=1)

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
    return (max(0, center - margin), min(1, center + margin))

qhat, pred_sets = conformal_binary(y_calib_final, y_calib, y_test_final, ALPHA)
set_sizes = pred_sets.sum(axis=1)
covered = pred_sets[np.arange(len(y_test)), y_test.astype(int)]
ci_lo, ci_hi = wilson_ci(covered.sum(), len(covered))

print(f"qhat: {qhat:.4f}")
print(f"Marginal coverage: {covered.mean():.4f}  95% CI=[{ci_lo:.3f},{ci_hi:.3f}]  "
      f"(target {1-ALPHA:.3f})")
print(f"Mean set size: {set_sizes.mean():.4f}  "
      f"(singleton:{(set_sizes==1).mean():.2%}  both:{(set_sizes==2).mean():.2%})")

print("\n[7/7] Saving...")
os.makedirs('/kaggle/working/models', exist_ok=True)
os.makedirs('/kaggle/working/results', exist_ok=True)
model.save('/kaggle/working/models/lstm_baseline.keras')
joblib.dump(scaler, '/kaggle/working/models/lstm_feature_scaler.pkl')
if calibrator_obj is not None:
    joblib.dump(calibrator_obj, f'/kaggle/working/models/lstm_calibrator_{best_calib}.pkl')

np.savez('/kaggle/working/results/lstm_baseline_results.npz',
         auc=auc, f1=f1, accuracy=acc, precision=prec, recall=rec, threshold=threshold,
         qhat=qhat, coverage=covered.mean(), coverage_ci=(ci_lo, ci_hi),
         mean_set_size=set_sizes.mean(), set_sizes=set_sizes, covered=covered,
         y_test_pred=y_test_pred, y_test_calibrated=y_test_final)

print("\nSaved:")
print("  /kaggle/working/models/lstm_baseline.keras")
print("  /kaggle/working/models/lstm_feature_scaler.pkl")
if calibrator_obj is not None:
    print(f"  /kaggle/working/models/lstm_calibrator_{best_calib}.pkl")
print("  /kaggle/working/results/lstm_baseline_results.npz")
print("\nCOMPARISON TABLE (fill in Transformer numbers from Stage 2/3 output):")
print(f"{'Metric':20s} {'LSTM':>10s} {'Transformer':>14s}")
print(f"{'AUC':20s} {auc:10.4f} {'--':>14s}")
print(f"{'F1':20s} {f1:10.4f} {'--':>14s}")
print(f"{'Marginal coverage':20s} {covered.mean():10.4f} {'--':>14s}")
print(f"{'Mean set size':20s} {set_sizes.mean():10.4f} {'--':>14s}")
print("STAGE 2B COMPLETE")
