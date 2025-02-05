import pandas as pd

df = pd.read_csv('data/4-2-25/position.csv')
df.insert(1, 'stateEstimate.x', 0)
df.insert(2, 'stateEstimate.y', 0)
df.to_csv('data/4-2-25/positionnew.csv', index=False)
print(df)
