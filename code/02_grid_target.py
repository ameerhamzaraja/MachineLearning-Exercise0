import pandas as pd

pd.set_option("display.max_columns", None)

df = pd.read_csv("data/Data_for_UCI_named.csv")

print(df.isna().sum().sum())
print(df["stabf"].value_counts())
print(df["stabf"].value_counts(normalize=True))
print(df["stab"].describe())

import matplotlib.pyplot as plt

counts = df["stabf"].value_counts()
plt.bar(counts.index, counts.values)
plt.title("Target stabf")
plt.ylabel("number of samples")
plt.savefig("figures/stabf_counts.png", dpi=200)
plt.show()

check = (df["stab"] < 0) == (df["stabf"] == "stable")
print(check.all())

plt.figure()
stable = df[df["stabf"] == "stable"]["stab"]
unstable = df[df["stabf"] == "unstable"]["stab"]
plt.hist([stable, unstable], bins=40, stacked=True, label=["stable", "unstable"])
plt.axvline(0, color="black", linestyle="--")
plt.xlabel("stab")
plt.ylabel("number of samples")
plt.legend()
plt.savefig("figures/stab_hist.png", dpi=200)
plt.show()

print((df["p1"] + df["p2"] + df["p3"] + df["p4"]).abs().max())
print(df[["tau1", "g1", "p1", "p2", "stab"]].corr().round(3))