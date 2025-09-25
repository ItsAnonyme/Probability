import pandas as pd

Data = pd.read_csv("Dataset-small.csv", skiprows=1, skipfooter=216, engine='python')

print(Data.columns)
print(Data.head())
