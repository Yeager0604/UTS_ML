"""
generate_dataset.py
Membuat dataset sintetis menyerupai Kaggle: joshmcadams/oranges-vs-grapefruit
Jalankan sekali sebelum classification.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 10000  # total sampel (5000 per kelas)

# ── Orange: diameter kecil, berat ringan, warna oranye terang ──
orange_df = pd.DataFrame({
    'diameter': np.random.normal(7.5,  0.5,  n // 2),
    'weight':   np.random.normal(150,  15,   n // 2),
    'red':      np.random.normal(220,  10,   n // 2),
    'green':    np.random.normal(120,  10,   n // 2),
    'blue':     np.random.normal(50,   10,   n // 2),
    'name':     'orange'
})

# ── Grapefruit: diameter besar, berat berat, warna lebih pucat ──
grape_df = pd.DataFrame({
    'diameter': np.random.normal(11.0, 0.7,  n // 2),
    'weight':   np.random.normal(350,  30,   n // 2),
    'red':      np.random.normal(200,  10,   n // 2),
    'green':    np.random.normal(150,  10,   n // 2),
    'blue':     np.random.normal(80,   10,   n // 2),
    'name':     'grapefruit'
})

df = (
    pd.concat([orange_df, grape_df])
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

df.to_csv('citrus.csv', index=False)
print("=" * 50)
print("  Dataset citrus.csv berhasil dibuat!")
print("=" * 50)
print(f"  Total sampel : {len(df):,}")
print(f"  Fitur        : {list(df.columns[:-1])}")
print(f"\n  Distribusi kelas:")
print(df['name'].value_counts().to_string())
print("=" * 50)
