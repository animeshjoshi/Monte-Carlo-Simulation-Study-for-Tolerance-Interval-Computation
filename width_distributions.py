import pandas as pd
import numpy as np

df = pd.read_csv('detailed_coverage_results.csv')

print(df.shape)

import numpy as np
import matplotlib.pyplot as plt

data = df['Pareto Distribution Non Parametric TI Widths']
# Threshold value
threshold = 0.95

# Compute statistics
mean_val = np.mean(data)


# Plot setup
plt.figure(figsize=(10,6))

# Histogram
counts, bins, patches = plt.hist(data, bins=30, color='salmon', edgecolor='black', alpha=0.7)

# Mean line
plt.axvline(mean_val, color='black', linestyle='--', linewidth=2, label=f"Mean = {mean_val:.2f}")


# Labels and title
plt.xlabel("Value", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.title("Distribution of Widths for Non-Parametric Pareto TI", fontsize=16, fontweight='bold')

# Legend
plt.legend(fontsize=12)

# Publication-ready layout
plt.tight_layout()

plt.show()