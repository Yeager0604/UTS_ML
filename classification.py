"""
classification.py
=================
Klasifikasi Jeruk (Orange) vs Grapefruit
Model  : Decision Tree | Naive Bayes | Support Vector Machine
Dataset: citrus.csv (jalankan generate_dataset.py terlebih dahulu)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

SEED = 42
LINE = "=" * 60

# ─────────────────────────────────────────────────────────────
# TAHAP 1 – MEMUAT DATASET
# ─────────────────────────────────────────────────────────────
print(LINE)
print("  TAHAP 1 : MEMUAT DATASET")
print(LINE)

df = pd.read_csv('citrus.csv')
print(f"Shape dataset : {df.shape}")
print(f"\nDistribusi kelas:\n{df['name'].value_counts().to_string()}")
print(f"\nStatistik deskriptif:")
print(df.describe().round(2).to_string())

# ─────────────────────────────────────────────────────────────
# TAHAP 2 – PREPROCESSING
# ─────────────────────────────────────────────────────────────
print(f"\n{LINE}")
print("  TAHAP 2 : PREPROCESSING")
print(LINE)

# Cek missing values
missing = df.isnull().sum()
print(f"Missing values:\n{missing.to_string()}")

# Encode label
le = LabelEncoder()
df['label'] = le.fit_transform(df['name'])   # grapefruit=0, orange=1
mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"\nMapping label : {mapping}")

# Fitur & target
FEATURES = ['diameter', 'weight', 'red', 'green', 'blue']
X = df[FEATURES]
y = df['label']

# Train-test split (80:20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"\nData train : {len(X_train):,} sampel  ({len(X_train)/len(df)*100:.0f}%)")
print(f"Data test  : {len(X_test):,}  sampel  ({len(X_test)/len(df)*100:.0f}%)")

# StandardScaler (untuk Naive Bayes & SVM)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────
# TAHAP 3 – EKSPLORASI DATA (EDA)
# ─────────────────────────────────────────────────────────────
print(f"\n{LINE}")
print("  TAHAP 3 : EKSPLORASI DATA (EDA)")
print(LINE)

COLORS = {'orange': '#FF8C00', 'grapefruit': '#DC143C'}

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Distribusi Fitur: Orange vs Grapefruit', fontsize=16, fontweight='bold')

for i, col in enumerate(FEATURES):
    ax = axes[i // 3][i % 3]
    for name, grp in df.groupby('name'):
        ax.hist(grp[col], bins=40, alpha=0.65, label=name, color=COLORS[name])
    ax.set_title(col, fontsize=12)
    ax.set_xlabel('Nilai')
    ax.set_ylabel('Frekuensi')
    ax.legend()
    ax.grid(alpha=0.3)

axes[1][2].axis('off')
plt.tight_layout()
plt.savefig('eda_distributions.png', dpi=120, bbox_inches='tight')
print("  Grafik EDA disimpan  → eda_distributions.png")

# Correlation heatmap
fig2, ax2 = plt.subplots(figsize=(7, 5))
corr = df[FEATURES + ['label']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax2, square=True)
ax2.set_title('Korelasi Antar Fitur', fontsize=13)
plt.tight_layout()
plt.savefig('eda_correlation.png', dpi=120, bbox_inches='tight')
print("  Grafik korelasi disimpan → eda_correlation.png")

# ─────────────────────────────────────────────────────────────
# TAHAP 4 – TRAINING & EVALUASI
# ─────────────────────────────────────────────────────────────
print(f"\n{LINE}")
print("  TAHAP 4 : TRAINING & EVALUASI MODEL")
print(LINE)

results = {}

def evaluate(model_name, model, X_tr, X_te, y_tr, y_te):
    """Latih model dan hitung metrik evaluasi."""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob)
    cv  = cross_val_score(model, X_tr, y_tr, cv=5, scoring='accuracy').mean()
    cm  = confusion_matrix(y_te, y_pred)
    rep = classification_report(y_te, y_pred, target_names=le.classes_)

    print(f"\n{'─'*55}")
    print(f"  🔹 {model_name}")
    print(f"{'─'*55}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  ROC-AUC     : {auc:.4f}")
    print(f"  CV (5-fold) : {cv:.4f}")
    print(f"\nClassification Report:\n{rep}")

    results[model_name] = {
        'model':  model,
        'acc':    acc,
        'auc':    auc,
        'cv':     cv,
        'cm':     cm,
        'y_pred': y_pred,
        'y_prob': y_prob,
    }
    return model

# 4a. Decision Tree
dt_model = evaluate(
    "Decision Tree",
    DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=SEED),
    X_train, X_test, y_train, y_test
)

# 4b. Naive Bayes
nb_model = evaluate(
    "Naive Bayes (GaussianNB)",
    GaussianNB(),
    X_train_sc, X_test_sc, y_train, y_test
)

# 4c. Support Vector Machine
svm_model = evaluate(
    "Support Vector Machine",
    SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=SEED),
    X_train_sc, X_test_sc, y_train, y_test
)

# ─────────────────────────────────────────────────────────────
# TAHAP 5 – PERBANDINGAN MODEL
# ─────────────────────────────────────────────────────────────
print(f"\n{LINE}")
print("  TAHAP 5 : PERBANDINGAN MODEL")
print(LINE)

summary = pd.DataFrame({
    'Model':        list(results.keys()),
    'Accuracy':     [v['acc'] for v in results.values()],
    'ROC-AUC':      [v['auc'] for v in results.values()],
    'CV (5-fold)':  [v['cv']  for v in results.values()],
}).set_index('Model')

print(summary.round(4).to_string())
best_model = summary['Accuracy'].idxmax()
print(f"\n  🏆 Model Terbaik : {best_model}")
print(f"     Accuracy      : {summary.loc[best_model, 'Accuracy']:.4f}")

# ─────────────────────────────────────────────────────────────
# TAHAP 6 – VISUALISASI HASIL
# ─────────────────────────────────────────────────────────────
print(f"\n{LINE}")
print("  TAHAP 6 : MENYIMPAN VISUALISASI")
print(LINE)

# --- Confusion Matrices ---
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Confusion Matrix – Ketiga Model', fontsize=14, fontweight='bold')

for ax, (name, v) in zip(axes, results.items()):
    sns.heatmap(
        v['cm'], annot=True, fmt='d', cmap='Blues',
        xticklabels=le.classes_, yticklabels=le.classes_, ax=ax,
        linewidths=0.5
    )
    ax.set_title(f"{name}\nAcc={v['acc']:.4f}", fontsize=11)
    ax.set_xlabel('Prediksi');  ax.set_ylabel('Aktual')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=120, bbox_inches='tight')
print("  Confusion matrices disimpan → confusion_matrices.png")

# --- ROC Curves ---
fig, ax = plt.subplots(figsize=(8, 6))
for name, v in results.items():
    fpr, tpr, _ = roc_curve(y_test, v['y_prob'])
    ax.plot(fpr, tpr, lw=2, label=f"{name}  (AUC={v['auc']:.4f})")
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve – Perbandingan Semua Model', fontsize=13)
ax.legend(fontsize=10);  ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=120, bbox_inches='tight')
print("  ROC curves disimpan      → roc_curves.png")

# --- Bar Chart Perbandingan ---
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(summary));  w = 0.25
metrics = ['Accuracy', 'ROC-AUC', 'CV (5-fold)']
palette = ['#4C72B0', '#DD8452', '#55A868']

for idx, (metric, color) in enumerate(zip(metrics, palette)):
    bars = ax.bar(x + (idx - 1) * w, summary[metric], w, label=metric, color=color)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{bar.get_height():.3f}",
            ha='center', va='bottom', fontsize=8
        )

ax.set_xticks(x);  ax.set_xticklabels(summary.index, rotation=10, fontsize=10)
ax.set_ylim(0.85, 1.03)
ax.set_title('Perbandingan Performa Model', fontsize=13)
ax.legend();  ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=120, bbox_inches='tight')
print("  Bar chart disimpan       → model_comparison.png")

# --- Decision Tree Visualization ---
fig, ax = plt.subplots(figsize=(22, 9))
plot_tree(
    dt_model, feature_names=FEATURES, class_names=le.classes_,
    filled=True, max_depth=3, fontsize=9, rounded=True, ax=ax
)
ax.set_title('Visualisasi Decision Tree (max_depth tampil = 3)', fontsize=14)
plt.tight_layout()
plt.savefig('decision_tree_viz.png', dpi=100, bbox_inches='tight')
print("  Decision tree disimpan   → decision_tree_viz.png")

# ─────────────────────────────────────────────────────────────
# TAHAP 7 – PREDIKSI DATA BARU
# ─────────────────────────────────────────────────────────────
print(f"\n{LINE}")
print("  TAHAP 7 : PREDIKSI BUAH BARU")
print(LINE)

samples = {
    "Contoh Buah A  (perkiraan: Orange)":
        [[7.5, 150, 220, 120, 50]],
    "Contoh Buah B  (perkiraan: Grapefruit)":
        [[11.0, 350, 200, 150, 80]],
    "Contoh Buah C  (ukuran tengah)":
        [[9.0, 250, 210, 135, 65]],
}

header = f"{'Buah':<42} {'Decision Tree':<18} {'Naive Bayes':<18} {'SVM'}"
print(header)
print("─" * len(header))

for desc, s in samples.items():
    s_sc   = scaler.transform(s)
    dt_p   = le.inverse_transform(dt_model.predict(s))[0]
    nb_p   = le.inverse_transform(nb_model.predict(s_sc))[0]
    svm_p  = le.inverse_transform(svm_model.predict(s_sc))[0]
    print(f"{desc:<42} {dt_p:<18} {nb_p:<18} {svm_p}")

print(f"\n{LINE}")
print("  ✅  Semua tahapan selesai!")
print(f"  File grafik tersimpan di folder yang sama.")
print(LINE)
