"""
Stage 3b: DeLong significance test for the Transformer vs LSTM AUC
comparison on the identical patient-grouped test set.
"""
import numpy as np
from scipy import stats

# ---- Fast DeLong implementation (Sun & Xu, 2014) ----
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2

def _fast_delong(preds_sorted_transposed, label_1_count):
    m = label_1_count
    n = preds_sorted_transposed.shape[1] - m
    positive = preds_sorted_transposed[:, :m]
    negative = preds_sorted_transposed[:, m:]
    k = preds_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov

def delong_roc_test(y_true, prob_a, prob_b):
    """Two-sided p-value for AUC(prob_a) == AUC(prob_b) on the SAME y_true
    (paired test set), via DeLong's method."""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true)  # positives first
    y_sorted = y_true[order]
    label_1_count = int(y_sorted.sum())

    preds = np.vstack([np.asarray(prob_a)[order], np.asarray(prob_b)[order]])
    aucs, delongcov = _fast_delong(preds, label_1_count)

    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(l.dot(delongcov).dot(l.T))
    p = 2 * (1 - stats.norm.cdf(z))
    return aucs[0], aucs[1], float(z), float(p.item())


if __name__ == "__main__":
    # ---- Load saved predictions (already on disk from Stages 2/2b/3) ----
    conf = np.load('/kaggle/working/results/conformal_results.npz')
    lstm = np.load('/kaggle/working/results/lstm_baseline_results.npz')

    y_test = conf['y_test']                     # Transformer's y_test (corrected split)
    p_test_transformer = conf['p_test']
    p_test_lstm = lstm['y_test_calibrated']

    assert len(y_test) == len(p_test_lstm), (
        "Length mismatch -- confirm the LSTM script used "
        "corrected_split_indices.npz, not the uncorrected split. "
        "If lengths differ, re-run Stage 2b after Stage 3 in the same "
        "session so it picks up the corrected indices."
    )

    auc_t, auc_l, z, p = delong_roc_test(y_test, p_test_transformer, p_test_lstm)
    print(f"Transformer AUC (DeLong): {auc_t:.4f}")
    print(f"LSTM AUC (DeLong):        {auc_l:.4f}")
    print(f"z = {z:.3f}, two-sided p = {p:.4f}")
    print("Significant at alpha=0.05" if p < 0.05 else "Not significant at alpha=0.05")

    np.savez('/kaggle/working/results/delong_test.npz',
             auc_transformer=auc_t, auc_lstm=auc_l, z=z, p=p)
