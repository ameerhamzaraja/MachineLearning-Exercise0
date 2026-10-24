import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

pd.set_option("display.max_columns", None)

path = "data/hr_analytics_job_change.arff"
with open(path) as f:
    lines = f.read().splitlines()

for i, line in enumerate(lines):
    if line.strip().upper() == "@DATA":
        data_start = i
        break

names = []
for line in lines[:data_start]:
    if line.upper().startswith("@ATTRIBUTE"):
        names.append(line.split()[1].strip("'"))

body = "\n".join(lines[data_start + 1:])
df = pd.read_csv(StringIO(body), names=names, quotechar="'", skipinitialspace=True)

print(df.shape)
print(df.head(3))
print(df["LookingForJobChange"].value_counts())
print(df.isna().sum())

print((df == "?").sum().sum())

df = df.replace("?", np.nan)

print(df.isna().sum().sum())
print(df.isna().any(axis=1).sum())

missing = df.isna().sum()
missing = missing[missing > 0].sort_values()
missing_pct = missing / len(df) * 100
print(missing_pct.round(1))

plt.figure()
plt.barh(missing_pct.index, missing_pct.values, color="gray")
plt.xlabel("% missing")
plt.title("Missing values per attribute")
plt.tight_layout()
plt.savefig("figures/hr_missing.png", dpi=200)
plt.show()

no = df[df["LookingForJobChange"] == "No"]["city_development_index"]
yes = df[df["LookingForJobChange"] == "Yes"]["city_development_index"]
print(no.mean(), yes.mean())

plt.figure()
plt.hist([no, yes], bins=30, stacked=True, label=["No", "Yes"])
plt.xlabel("city_development_index")
plt.ylabel("number of samples")
plt.legend()
plt.savefig("figures/hr_cdi.png", dpi=200)
plt.show()

print(df["training_hours"].describe())
print(df["training_hours"].skew())

plt.figure()
plt.hist(df["training_hours"], bins=40, color="gray")
plt.axvline(df["training_hours"].median(), color="black", linestyle="--")
plt.xlabel("training_hours")
plt.ylabel("number of samples")
plt.savefig("figures/hr_hours.png", dpi=200)
plt.show()

order = ["<1"] + [str(i) for i in range(1, 21)] + [">20"]
counts = df["experience"].value_counts().reindex(order)
print(counts[">20"], counts["<1"])

plt.figure()
plt.bar(range(len(order)), counts.values)
plt.xticks([0, 5, 10, 15, 21], ["<1", "5", "10", "15", ">20"])
plt.xlabel("experience [years]")
plt.ylabel("number of samples")
plt.savefig("figures/hr_experience.png", dpi=200)
plt.show()

size_order = ["<10", "10/49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"]
size_counts = df["company_size"].value_counts().reindex(size_order)
n_missing = df["company_size"].isna().sum()
print(size_counts.to_dict(), n_missing)

plt.figure()
plt.bar(range(len(size_order)), size_counts.values, color="gray")
plt.bar([len(size_order)], [n_missing], color="lightgray")
plt.xticks(range(len(size_order) + 1), size_order + ["missing"], rotation=60)
plt.ylabel("number of samples")
plt.tight_layout()
plt.savefig("figures/hr_company_size.png", dpi=200)
plt.show()


print(df["city"].nunique())
print(df["city"].value_counts().head(3))
print((df["city"].value_counts() < 10).sum())
print(df.groupby("city")["city_development_index"].nunique().max())

wants = df["LookingForJobChange"] == "Yes"
print(wants[df["company_type"].isna()].mean(), wants[df["company_type"].notna()].mean())