import pandas as pd

# Load CSV
df = pd.read_csv("detailed_coverage_results.csv")

# Dictionary of replacements: "find_this" : "replace_with_this"
replacements = {
    "Normal TI": "T1",
    "Bootstrap TI": "T2",
    "Non Parametric TI": "T3",
    "KDE Bootstrap TI": "T4",
    "Smooth KDE Bootstrap TI": "T4",
    "Coverage": "O1",
    "Coverages": "O1",
    "Width": "O2",
    "Widths": "O2",
    "Normal Distribution": "B1",
    "Chi-Square Distribution": "B2",
    "F Distribution": "B3",
    "Log Normal Distribution": "B4",
    "T Distribution": "B5",
    "Gamma Distribution": "B6",
    "Beta Distribution": "B7",
    "Exponential Distribution": "B8",
    "Pareto Distribution": "B9"
}

# Apply replacements to all column names
new_columns = []
for col in df.columns:
    new_col = col
    for find, replace in replacements.items():
        if find in new_col:
            new_col = new_col.replace(find, replace)
    new_columns.append(new_col)

# Assign new column names
df.columns = new_columns

# Save the updated CSV
df.to_csv("updated_columns.csv", index=False)

print("Columns after find-and-replace:")
print(df.columns.tolist())
