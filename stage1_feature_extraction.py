import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

print("STAGE 1: MULTIMODAL FEATURE EXTRACTION")

mimic_path = '/kaggle/input/datasets/piyushtrivedi39/mimiciv/'

TARGET_COHORT_SIZE = 8000  # set to None to use every eligible stay with no cap

FEATURES = {
    'HR':       {'itemids': [220045],                    'table': 'chart', 'low': 0,   'high': 350},
    'RR':       {'itemids': [220210, 224690],             'table': 'chart', 'low': 0,   'high': 70},
    'SBP':      {'itemids': [220050, 220179, 225309],     'table': 'chart', 'low': 0,   'high': 400},
    'DBP':      {'itemids': [220051, 220180, 225310],     'table': 'chart', 'low': 0,   'high': 300},
    'MBP':      {'itemids': [220052, 220181, 225312],     'table': 'chart', 'low': 0,   'high': 300},
    'SpO2':     {'itemids': [220277],                     'table': 'chart', 'low': 0,   'high': 100},
    'TempC':    {'itemids': [223762],                     'table': 'chart', 'low': 10,  'high': 50},
    'TempF':    {'itemids': [223761],                     'table': 'chart', 'low': 70,  'high': 120},
    'Glucose':  {'itemids': [220621, 225664, 226537],     'table': 'chart', 'low': 0,   'high': 2000},
    'FiO2':     {'itemids': [223835],                     'table': 'chart', 'low': 0.2, 'high': 1.0},
}

LAB_FEATURES = {
    'Creatinine': {'itemids': [50912], 'low': 0,   'high': 150},
    'Lactate':    {'itemids': [50813], 'low': 0,   'high': 50},
    'Hgb':        {'itemids': [50811], 'low': 0,   'high': 50},
    'WBC':        {'itemids': [51301], 'low': 0,   'high': 1000},
    'Sodium':     {'itemids': [50983], 'low': 100, 'high': 200},
    'Potassium':  {'itemids': [50971], 'low': 0,   'high': 15},
    'Bilirubin':  {'itemids': [50885], 'low': 0,   'high': 150},
}

N_FEATURES = len(FEATURES) + len(LAB_FEATURES)
N_HOURS = 48

print(f"Feature set: {N_FEATURES} features x {N_HOURS} hours")
print(f"  Chart: {list(FEATURES.keys())}")
print(f"  Labs:  {list(LAB_FEATURES.keys())}")

print("\n[1/5] Building cohort...")
patients = pd.read_csv(f'{mimic_path}patients.csv')
admissions = pd.read_csv(f'{mimic_path}admissions.csv',
                         parse_dates=['admittime','dischtime','edregtime','edouttime'])
icustays = pd.read_csv(f'{mimic_path}icustays.csv', parse_dates=['intime','outtime'])

icu = icustays.merge(admissions[['hadm_id','hospital_expire_flag']], on='hadm_id')
icu = icu.merge(patients[['subject_id','anchor_age']], on='subject_id')
icu = icu.merge(patients[['subject_id','gender']], on='subject_id')
icu['los_hours'] = (icu['outtime'] - icu['intime']).dt.total_seconds() / 3600
icu = icu[(icu['los_hours'] >= 48) & (icu['anchor_age'] >= 18)]

print(f"Eligible stays before cap (los>=48h, age>=18): {len(icu)}")

if TARGET_COHORT_SIZE is not None and len(icu) > TARGET_COHORT_SIZE:
    icu, _ = train_test_split(icu, train_size=TARGET_COHORT_SIZE, random_state=42,
                              stratify=icu['hospital_expire_flag'])
elif TARGET_COHORT_SIZE is not None and len(icu) < TARGET_COHORT_SIZE:
    print(f"WARNING: only {len(icu)} eligible stays exist; requested target of "
          f"{TARGET_COHORT_SIZE} is unreachable. Using all {len(icu)} eligible stays.")

print(f"Cohort: {len(icu)} stays, mortality: {icu['hospital_expire_flag'].mean():.2%}")
icu = icu.reset_index(drop=True)
cohort_set = set(zip(icu['subject_id'], icu['hadm_id']))

all_chart_ids = set(id_ for f in FEATURES.values() for id_ in f['itemids'])
all_lab_ids   = set(id_ for f in LAB_FEATURES.values() for id_ in f['itemids'])

print("\n[2/5] Loading chartevents (valuenum only)...")
chart_chunks = []
for chunk in pd.read_csv(f'{mimic_path}chartevents.csv', chunksize=100000,
                         usecols=['subject_id','hadm_id','itemid','charttime','valuenum']):
    chunk = chunk[chunk['itemid'].isin(all_chart_ids)]
    chunk = chunk.set_index(['subject_id','hadm_id'])
    chunk = chunk[chunk.index.isin(cohort_set)]
    chart_chunks.append(chunk.reset_index())

chartevents = pd.concat(chart_chunks, ignore_index=True) if chart_chunks else pd.DataFrame(
    columns=['subject_id','hadm_id','itemid','charttime','valuenum'])
chartevents['charttime'] = pd.to_datetime(chartevents['charttime'])
chartevents = chartevents.dropna(subset=['valuenum'])
print(f"Chartevents: {len(chartevents)} rows for {chartevents[['subject_id','hadm_id']].drop_duplicates().shape[0]} patients")

print("\n[3/5] Loading labevents...")
lab_chunks = []
for chunk in pd.read_csv(f'{mimic_path}labevents.csv', chunksize=100000,
                         usecols=['subject_id','hadm_id','itemid','charttime','valuenum']):
    chunk = chunk[chunk['itemid'].isin(all_lab_ids)]
    chunk = chunk.set_index(['subject_id','hadm_id'])
    chunk = chunk[chunk.index.isin(cohort_set)]
    lab_chunks.append(chunk.reset_index())

labevents = pd.concat(lab_chunks, ignore_index=True) if lab_chunks else pd.DataFrame(
    columns=['subject_id','hadm_id','itemid','charttime','valuenum'])
labevents['charttime'] = pd.to_datetime(labevents['charttime'])
labevents = labevents.dropna(subset=['valuenum'])
print(f"Labevents: {len(labevents)} rows for {labevents[['subject_id','hadm_id']].drop_duplicates().shape[0]} patients")

print("\n[4/5] Per-feature hourly timeseries (vectorized, benchmark-aligned)...")

feature_names = list(FEATURES.keys()) + list(LAB_FEATURES.keys())
feat_to_idx   = {f: i for i, f in enumerate(feature_names)}
tempf_idx     = feat_to_idx.get('TempF', -1)
tempc_idx     = feat_to_idx.get('TempC', -1)

n_patients = len(icu)
# (subject_id, hadm_id) is not unique in MIMIC-IV (a single admission can contain
# multiple ICU stays), so events are joined via merge rather than a dict/index keyed
# on that pair; the merge broadcasts each event to every matching icustay.
icu_keyed = icu[['subject_id', 'hadm_id', 'intime']].reset_index(drop=True).reset_index().rename(columns={'index': 'pidx'})

X = np.full((n_patients, N_HOURS, N_FEATURES), np.nan, dtype='float64')

itemid_to_feat, itemid_to_low, itemid_to_high = {}, {}, {}
for fname, meta in {**FEATURES, **LAB_FEATURES}.items():
    for iid in meta['itemids']:
        itemid_to_feat[iid] = fname
        itemid_to_low[iid]  = meta['low']
        itemid_to_high[iid] = meta['high']


def ffill_bfill_2d(arr):
    """Forward-fill then back-fill NaNs along axis=1 (hours), vectorized across all patients."""
    n_p, n_h = arr.shape
    mask = ~np.isnan(arr)

    idx = np.where(mask, np.arange(n_h), -1)
    idx = np.maximum.accumulate(idx, axis=1)
    filled_fwd = np.where(idx >= 0, arr[np.arange(n_p)[:, None], np.clip(idx, 0, None)], np.nan)

    rev = filled_fwd[:, ::-1]
    mask_rev = ~np.isnan(rev)
    idx_rev = np.where(mask_rev, np.arange(n_h), -1)
    idx_rev = np.maximum.accumulate(idx_rev, axis=1)
    filled_rev = np.where(idx_rev >= 0, rev[np.arange(n_p)[:, None], np.clip(idx_rev, 0, None)], np.nan)
    return filled_rev[:, ::-1]


for events_df in [chartevents, labevents]:
    if len(events_df) == 0:
        continue
    df = events_df.copy()
    df = df[df['itemid'].isin(itemid_to_feat.keys())]
    if len(df) == 0:
        continue
    df['feature'] = df['itemid'].map(itemid_to_feat)
    df['low']     = df['itemid'].map(itemid_to_low)
    df['high']    = df['itemid'].map(itemid_to_high)
    df['valuenum'] = np.clip(df['valuenum'].values, df['low'].values, df['high'].values)

    df = df.merge(icu_keyed, on=['subject_id', 'hadm_id'], how='inner')
    df['hour'] = ((df['charttime'] - df['intime']).dt.total_seconds() / 3600).astype(int)
    df = df[(df['hour'] >= 0) & (df['hour'] < N_HOURS)]
    df['fidx'] = df['feature'].map(feat_to_idx)

    grouped = df.groupby(['pidx', 'hour', 'fidx'])['valuenum'].mean().reset_index()
    X[grouped['pidx'].values, grouped['hour'].values, grouped['fidx'].values] = grouped['valuenum'].values

miss_list = np.isnan(X).mean(axis=(1, 2))

if tempf_idx >= 0 and tempc_idx >= 0:
    tempf_col = X[:, :, tempf_idx]
    tempc_col = X[:, :, tempc_idx]
    has_f = ~np.isnan(tempf_col)
    converted = (tempf_col - 32.0) * 5.0 / 9.0
    fill_mask = has_f & np.isnan(tempc_col)
    tempc_col[fill_mask] = converted[fill_mask]
    X[:, :, tempc_idx] = tempc_col
    X[:, :, tempf_idx] = np.nan

# captured before fill so imputed values aren't counted as observed
observed_mask = ~np.isnan(X)

for fi in range(N_FEATURES):
    X[:, :, fi] = ffill_bfill_2d(X[:, :, fi])

X = np.nan_to_num(X, nan=0.0).astype('float32')
y = icu['hospital_expire_flag'].to_numpy(dtype='float32')
subject_id = icu['subject_id'].to_numpy()
hadm_id    = icu['hadm_id'].to_numpy()

print("\n[5/5] Validation...")
print(f"Shape: {X.shape}  (patients x hours x features)")
print(f"Mortality: {y.mean():.2%}")
print(f"Mean missingness before fill: {np.mean(miss_list):.2%}")
print("\nFeature stats (non-zero values):")
for i, fn in enumerate(feature_names):
    nz = X[:,:,i][X[:,:,i] != 0]
    if len(nz):
        print(f"  {fn:12s}  min={nz.min():.1f}  max={nz.max():.1f}  mean={nz.mean():.1f}")
    else:
        print(f"  {fn:12s}  --- ALL ZERO (no data) ---")

n_unique_subjects = len(np.unique(subject_id))
n_repeat_subjects = len(subject_id) - n_unique_subjects
print(f"Unique patients (subject_id): {n_unique_subjects}  |  "
      f"stays sharing a subject_id with another stay: {n_repeat_subjects}")

os.makedirs('/kaggle/working', exist_ok=True)
np.save('/kaggle/working/X.npy', X)
np.save('/kaggle/working/y.npy', y)
np.save('/kaggle/working/feature_names.npy', np.array(feature_names))
np.save('/kaggle/working/subject_id.npy', subject_id)
np.save('/kaggle/working/hadm_id.npy', hadm_id)
np.save('/kaggle/working/observed_mask.npy', observed_mask)
np.save('/kaggle/working/los_hours.npy', icu['los_hours'].to_numpy())
np.save('/kaggle/working/age.npy', icu['anchor_age'].to_numpy())
np.save('/kaggle/working/gender.npy', icu['gender'].to_numpy())

print(f"\nSTAGE 1 COMPLETE  X={X.shape}  y={y.shape}")
print("Saved: X.npy, y.npy, feature_names.npy, subject_id.npy, hadm_id.npy, observed_mask.npy, los_hours.npy, age.npy, gender.npy")
