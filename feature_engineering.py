import pandas as pd
import numpy as np

df = pd.read_csv('detailed_coverage_results.csv')

print(df.shape)

for x in df.columns:

    print(x)