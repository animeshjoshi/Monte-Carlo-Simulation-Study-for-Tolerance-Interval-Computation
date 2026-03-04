import pandas as pd
import numpy as np

df = pd.read_csv('detailed_coverage_results.csv')

print(df.shape)

import numpy as np
import matplotlib.pyplot as plt

data = df['Pareto Distribution Smooth KDE Bootstrap TI Coverages']
# Threshold value
threshold = 0.95

# Compute statistics
mean_val = np.mean(data)
pct_above = np.mean(data >= threshold) * 100

# Plot setup
plt.figure(figsize=(10,6))

# Histogram
counts, bins, patches = plt.hist(data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)

# Mean line
plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f"Mean = {mean_val:.2f}")

# Threshold line
plt.axvline(threshold, color='green', linestyle='-', linewidth=2, label=f"% of Values Over Threshold = {round(pct_above,1)}")


# Labels and title
plt.xlabel("Value", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.title("Distribution of Coverages for Bootstrap Smooth KDE Pareto TI", fontsize=16, fontweight='bold')

# Legend
plt.legend(fontsize=12)

# Publication-ready layout
plt.tight_layout()

plt.show()