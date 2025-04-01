import pandas as pd
import os

def readcsv(whole_path):
    path, filename = whole_path.rsplit('/', 1)
    path = path+'/'
    # Load the CSV files
    df1 = pd.read_csv(path+'battery.csv')  # Columns: [timestamp, col1]
    df2 = pd.read_csv(path+'motor.csv')  # Columns: [timestamp, col1, col2, col3, col4]

    if not os.path.exists(path+'position.csv'):
        df3 = df1.copy()
        df3 = df1.loc[:, ~df1.columns.str.contains('pm.vbat')]
        df3[['stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z']] = 0
    else:
        df3 = pd.read_csv(path+'position.csv')

    df1 = df1.loc[:, ~df1.columns.str.contains('Unnamed')]  # Drops 'Unnamed: 2'
    df2 = df2.loc[:, ~df2.columns.str.contains('Unnamed')]  # Drops 'Unnamed: 5'
    df3 = df3.loc[:, ~df3.columns.str.contains('Unnamed')]  # Drops 'Unnamed: 5'

    # Rename timestamp columns for consistency (if needed)
    df1 = df1.rename(columns={df1.columns[0]: 'timestamp'})
    df2 = df2.rename(columns={df2.columns[0]: 'timestamp'})
    df3 = df3.rename(columns={df3.columns[0]: 'timestamp'})

    # --- Inspect raw data ---
    # print("\n=== Raw Data Properties ===")
    # print(f"df1 shape: {df1.shape} | Columns: {list(df1.columns)}")
    # print(f"df2 shape: {df2.shape} | Columns: {list(df2.columns)}")
    # print(f"df3 shape: {df3.shape} | Columns: {list(df3.columns)}")
    # print("\nSample from df1 (first 3 rows):")
    # print(df1.head(3))
    # print("\nSample from df2 (first 3 rows):")
    # print(df2.head(3))
    # print("\nSample from df3 (first 3 rows):")
    # print(df3.head(3))

    # --- Round timestamps to nearest 100 ms ---
    df1['timestamp_rounded'] = (df1['timestamp'] / 100).round() * 100
    df2['timestamp_rounded'] = (df2['timestamp'] / 100).round() * 100
    df3['timestamp_rounded'] = (df3['timestamp'] / 100).round() * 100

    # --- Merge on rounded timestamps ---
    df_merged = pd.merge(
        df1,
        df2,
        on='timestamp_rounded',
        how='inner',
        suffixes=('_pm', '_motor')
    )

    # Now merge with df3
    df_merged = pd.merge(
        df_merged,
        df3,
        on='timestamp_rounded',
        how='inner',
        suffixes=('', '_pos')  # Add suffix only if column names clash
    )

    # Drop the rounded column (optional)
    df_merged = df_merged.drop(columns=['timestamp_rounded'])

    # Sort by original timestamp from df1
    df_merged = df_merged.sort_values('timestamp_pm').reset_index(drop=True)

    # --- Inspect results ---
    # print("\n=== Merged Data (Aligned to 100 ms Intervals) ===")
    # print(f"Shape: {df_merged.shape}")
    # print("\nColumns:", df_merged.columns.tolist())
    # print("\nFirst 3 Rows:")
    # print(df_merged.head(20))

    if filename=='battery.csv':
        return df_merged[['timestamp_pm', 'pm.vbat']].copy()
    elif filename=='motor.csv':
        return df_merged[['timestamp_pm', 'motor.m1', 'motor.m2', 'motor.m3', 'motor.m4']]
    elif filename=='position.csv':
        return df_merged[['timestamp_pm', 'stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z']]

#print(readcsv('data/21-2-25/battery.csv'))

# Uncomment to save
# df_merged.to_csv('merged_aligned.csv', index=False)
