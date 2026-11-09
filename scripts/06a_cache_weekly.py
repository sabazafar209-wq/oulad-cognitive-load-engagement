import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from repo root
import pandas as pd
import time
t0 = time.time()
print("Reading raw clickstream...", flush=True)
vle = pd.read_csv('data/raw/studentVle.csv')
print(f"Loaded {len(vle)} rows in {time.time()-t0:.1f}s", flush=True)

key = ['id_student', 'code_module', 'code_presentation']
vle['week'] = (vle['date'] // 7).astype(int)

t1 = time.time()
weekly = vle.groupby(key + ['week']).agg(
    week_clicks=('sum_click', 'sum'),
    week_sites=('id_site', 'nunique'),
    week_events=('sum_click', 'count'),
).reset_index()
print(f"Weekly groupby done in {time.time()-t1:.1f}s -> {len(weekly)} rows", flush=True)

weekly.to_csv('data/oulad_weekly_clicks.csv', index=False)
print(f"Saved. Total time {time.time()-t0:.1f}s", flush=True)
