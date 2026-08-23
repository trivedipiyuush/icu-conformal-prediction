import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, precision_score,
                              recall_score, brier_score_loss, roc_curve, precision_recall_curve)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import joblib

np.random.seed(42)
tf.random.set_seed(42)

print("STAGE 2: MULTIMODAL FUSION TRANSFORMER")

print("\n[1/6] Loading features...")
X = np.load('/kaggle/working/X.npy')
y = np.load('/kaggle/working/y.npy')
feature_names = np.load('/kaggle/working/feature_names.npy', allow_pickle=True).tolist()

n_patients, n_hours, n_features = X.shape
print(f"X: {X.shape}  |  y: {y.shape}  |  mortality: {y.mean():.2%}")

VITALS_IDX = [i for i,f in enumerate(feature_names) if f in
              ['HR','RR','SBP','DBP','MBP','SpO2','TempC','Glucose','FiO2']]
LABS_IDX   = [i for i,f in enumerate(feature_names) if f in
              ['Creatinine','Lactate','Hgb','WBC','Sodium','Potassium','Bilirubin']]
print(f"Vitals features ({len(VITALS_IDX)}): {[feature_names[i] for i in VITALS_IDX]}")
print(f"Lab features ({len(LABS_IDX)}):   {[feature_names[i] for i in LABS_IDX]}")

print("\n[2/6] Normalizing per-feature...")
scaler = StandardScaler()
X_2d   = X.reshape(-1, n_features)
X_2d   = scaler.fit_transform(X_2d)
X      = X_2d.reshape(n_patients, n_hours, n_features).astype('float32')

print("\n[3/6] Stratified 60/20/20 split...")
idx_all = np.arange(n_patients)
idx_train, idx_tmp, y_train, y_tmp = train_test_split(
    idx_all, y, test_size=0.4, random_state=42, stratify=y)
idx_calib, idx_test, y_calib, y_test = train_test_split(
    idx_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
X_train, X_calib, X_test = X[idx_train], X[idx_calib], X[idx_test]
print(f"Train: {X_train.shape[0]} ({y_train.mean():.2%})  "
      f"Calib: {X_calib.shape[0]} ({y_calib.mean():.2%})  "
      f"Test: {X_test.shape[0]} ({y_test.mean():.2%})")

np.savez('/kaggle/working/split_indices.npz',
         train_idx=idx_train, calib_idx=idx_calib, test_idx=idx_test)

pos_weight   = (1 - y_train.mean()) / (y_train.mean() + 1e-6)
class_weight = {0: 1.0, 1: float(pos_weight)}
print(f"Class weight (positive): {class_weight[1]:.2f}")

print("\n[4/6] Building Multimodal Fusion Transformer...")

def transformer_encoder(inputs, n_heads, key_dim, ff_dim, dropout=0.1):
    attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=key_dim, dropout=dropout)(inputs, inputs)
    x    = layers.Add()([inputs, attn])
    x    = layers.LayerNormalization(epsilon=1e-6)(x)
    ff   = layers.Dense(ff_dim, activation='relu')(x)
    ff   = layers.Dropout(dropout)(ff)
    ff   = layers.Dense(inputs.shape[-1])(ff)
    x    = layers.Add()([x, ff])
    x    = layers.LayerNormalization(epsilon=1e-6)(x)
    return x

@tf.keras.utils.register_keras_serializable(package="fusion_transformer")
class FeatureGather(layers.Layer):
    """Selects a fixed subset of feature channels along axis=2.

    A Layer subclass rather than layers.Lambda, so the model can be
    saved/reloaded via the standard Keras deserialization path.
    """
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


inputs = layers.Input(shape=(n_hours, n_features))

vitals_in = FeatureGather(VITALS_IDX, name="vitals_gather")(inputs)
labs_in   = FeatureGather(LABS_IDX,   name="labs_gather")(inputs)

vitals_proj = layers.Dense(64)(vitals_in)
labs_proj   = layers.Dense(64)(labs_in)

vitals_enc = transformer_encoder(vitals_proj, n_heads=4, key_dim=16, ff_dim=128)
vitals_enc = transformer_encoder(vitals_enc,  n_heads=4, key_dim=16, ff_dim=128)

labs_enc   = transformer_encoder(labs_proj,   n_heads=4, key_dim=16, ff_dim=128)
labs_enc   = transformer_encoder(labs_enc,    n_heads=4, key_dim=16, ff_dim=128)

vitals_pool = layers.GlobalAveragePooling1D()(vitals_enc)
labs_pool   = layers.GlobalAveragePooling1D()(labs_enc)

fused = layers.Concatenate()([vitals_pool, labs_pool])

x = layers.Dense(64, activation='relu')(fused)
x = layers.Dropout(0.3)(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dropout(0.2)(x)
output = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs=inputs, outputs=output)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    metrics=[tf.keras.metrics.AUC(name='auc')]
)

n_params = model.count_params()
print(f"Model parameters: {n_params:,}")
print(f"Architecture: Dual-stream Transformer (vitals | labs) -> concat -> classify")

print("\n[5/6] Training...")
early_stop = callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=15,
                                     restore_best_weights=True, verbose=1)
reduce_lr  = callbacks.ReduceLROnPlateau(monitor='val_auc', mode='max', factor=0.5,
                                         patience=5, min_lr=1e-6, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_calib, y_calib),
    epochs=150,
    batch_size=64,
    class_weight=class_weight,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

print(f"Stopped at epoch {len(history.history['loss'])}")
print(f"Best val_auc: {max(history.history['val_auc']):.4f}")

print("\n[6/6] Evaluating on test set...")
y_test_pred  = model.predict(X_test,  verbose=0).flatten()
y_calib_pred = model.predict(X_calib, verbose=0).flatten()

# calibration is fit on the calib set only, never on test
iso_calibrator = IsotonicRegression(out_of_bounds='clip')
iso_calibrator.fit(y_calib_pred, y_calib)

platt_calibrator = LogisticRegression()
platt_calibrator.fit(y_calib_pred.reshape(-1, 1), y_calib)

y_calib_pred_iso   = iso_calibrator.predict(y_calib_pred)
y_test_pred_iso    = iso_calibrator.predict(y_test_pred)

y_calib_pred_platt = platt_calibrator.predict_proba(y_calib_pred.reshape(-1, 1))[:, 1]
y_test_pred_platt  = platt_calibrator.predict_proba(y_test_pred.reshape(-1, 1))[:, 1]

brier_raw   = brier_score_loss(y_calib, y_calib_pred)
brier_iso   = brier_score_loss(y_calib, y_calib_pred_iso)
brier_platt = brier_score_loss(y_calib, y_calib_pred_platt)
print(f"Brier score (calib) — raw: {brier_raw:.4f} | isotonic: {brier_iso:.4f} | platt: {brier_platt:.4f}")

calib_scores = {'raw': brier_raw, 'isotonic': brier_iso, 'platt': brier_platt}
best_calib = min(calib_scores, key=calib_scores.get)
print(f"Selected calibration method: {best_calib}")

if best_calib == 'isotonic':
    y_calib_final, y_test_final = y_calib_pred_iso, y_test_pred_iso
    calibrator_obj = iso_calibrator
elif best_calib == 'platt':
    y_calib_final, y_test_final = y_calib_pred_platt, y_test_pred_platt
    calibrator_obj = platt_calibrator
else:
    y_calib_final, y_test_final = y_calib_pred, y_test_pred
    calibrator_obj = None

# threshold: F1-optimal on calib set, using calibrated probabilities
precisions, recalls, pr_thresholds = precision_recall_curve(y_calib, y_calib_final)
f1_scores = np.divide(
    2 * precisions * recalls, precisions + recalls,
    out=np.zeros_like(precisions), where=(precisions + recalls) != 0
)
best_idx = np.argmax(f1_scores[:-1])  # last PR point has no corresponding threshold
threshold = pr_thresholds[best_idx]
print(f"Derived threshold (F1-optimal on calib set): {threshold:.4f}  "
      f"(calib F1 at this threshold: {f1_scores[best_idx]:.4f})")

fpr, tpr, roc_thresholds = roc_curve(y_calib, y_calib_final)
youden_idx = np.argmax(tpr - fpr)
youden_threshold = roc_thresholds[youden_idx]
print(f"Alternative threshold (Youden's J on calib set): {youden_threshold:.4f}")

# apply derived threshold to calibrated test predictions
auc  = roc_auc_score(y_test, y_test_pred)  # AUC is invariant to monotonic calibration
f1   = f1_score(y_test, (y_test_final > threshold).astype(int), zero_division=0)
acc  = accuracy_score(y_test, (y_test_final > threshold).astype(int))
prec = precision_score(y_test, (y_test_final > threshold).astype(int), zero_division=0)
rec  = recall_score(y_test, (y_test_final > threshold).astype(int), zero_division=0)
brier_test = brier_score_loss(y_test, y_test_final)

print("TEST SET RESULTS")
print(f"AUC:            {auc:.4f}  {'OK' if auc > 0.70 else ('MARGINAL' if auc > 0.65 else 'LOW')}")
print(f"F1:             {f1:.4f}")
print(f"Accuracy:       {acc:.4f}")
print(f"Precision:      {prec:.4f}")
print(f"Recall:         {rec:.4f}")
print(f"Brier (test):   {brier_test:.4f}")
print(f"Threshold used: {threshold:.4f} ({best_calib} calibration)")
print(f"Pred range: [{y_test_final.min():.3f}, {y_test_final.max():.3f}], "
      f"mean={y_test_final.mean():.3f}")

os.makedirs('/kaggle/working/models',   exist_ok=True)
os.makedirs('/kaggle/working/results',  exist_ok=True)

model.save('/kaggle/working/models/fusion_transformer.keras')
joblib.dump(scaler, '/kaggle/working/models/feature_scaler.pkl')
if calibrator_obj is not None:
    joblib.dump(calibrator_obj, f'/kaggle/working/models/calibrator_{best_calib}.pkl')
joblib.dump({'threshold': threshold, 'method': best_calib},
            '/kaggle/working/models/threshold_config.pkl')

np.savez('/kaggle/working/predictions.npz',
         y_calib=y_calib, y_calib_pred=y_calib_pred, y_calib_pred_calibrated=y_calib_final,
         y_test=y_test,   y_test_pred=y_test_pred,   y_test_pred_calibrated=y_test_final,
         threshold=threshold)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['loss'],     label='Train', alpha=0.8)
axes[0].plot(history.history['val_loss'], label='Calib', alpha=0.8)
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history['auc'],     label='Train', alpha=0.8)
axes[1].plot(history.history['val_auc'], label='Calib', alpha=0.8)
axes[1].set_title('AUC')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/results/training.png', dpi=100, bbox_inches='tight')
plt.close()

print("\nSaved:")
print("  /kaggle/working/models/fusion_transformer.keras")
print("  /kaggle/working/models/feature_scaler.pkl")
if calibrator_obj is not None:
    print(f"  /kaggle/working/models/calibrator_{best_calib}.pkl")
print("  /kaggle/working/models/threshold_config.pkl")
print("  /kaggle/working/split_indices.npz")
print("  /kaggle/working/predictions.npz")
print("  /kaggle/working/results/training.png")
print("TRAINING COMPLETE")
