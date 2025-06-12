import pandas as pd
from pathlib import Path

# ─── base directory & paths ─────────────────────────────────────────────────
BASE      = Path.cwd()
csv1_path = BASE / 'csv_file' / 'breast-level_annotations.csv'
csv2_path = BASE / 'csv_file' / 'finding_annotations.csv'
csv3_path = BASE / 'csv_file' / 'train.csv'   # contains patient_id
# ─────────────────────────────────────────────────────────────────────────────

# 1) Load CSVs
df1 = pd.read_csv(csv1_path)
df2 = pd.read_csv(csv2_path)
df3 = pd.read_csv(csv3_path, dtype={'image_id': str})

# 2) Binarize BI-RADS → new 'birads' column
def binarize_birads(df):
    df = df.copy()
    df['birads'] = (
        df['breast_birads']
          .str.extract(r'(\d+)', expand=False)
          .astype(int)
          .ge(4)
          .astype(int)
    )
    return df.drop(columns=['breast_birads'])

df1 = binarize_birads(df1)
df2 = binarize_birads(df2)

# 3) Merge the first two on ['study_id','image_id','birads']
df_meta = df1.merge(
    df2,
    on=['study_id','image_id','birads'],
    how='outer'
)

# 4) Rename patient_id→study_id in df3 so we can pull it in
df3 = df3.rename(columns={'patient_id':'new_study_id'})

# 5) Merge df_meta with df3 on image_id
df_main = df_meta.merge(
    df3,
    on='image_id',
    how='left'
)

# 6) Now overwrite the old study_id with the patient ID
df_main['study_id'] = df_main['new_study_id']

# 7) Drop the helper column
df_main = df_main.drop(columns=['new_study_id'])

# 8) Write out
output_path = BASE / 'main.csv'
df_main.to_csv(output_path, index=False)
print(f"Wrote merged data (with study_id=set to patient_id) → {output_path}")
