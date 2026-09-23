"""Phase 1 EDA for CALiSol-23 (DEC holdout). Run: python3 eda.py"""
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

D = Path(__file__).parent / "ca-li-sol-23-challenge"
FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(exist_ok=True)
pd.set_option("display.width", 200, "display.max_columns", 50, "display.max_colwidth", 200)

tr = pd.read_csv(D / "train.csv")
te = pd.read_csv(D / "test.csv")
ss = pd.read_csv(D / "sample_submission.csv")
sp = pd.read_csv(D / "solvent_properties.csv")
meta = pd.read_csv(D / "metaData.csv")


def h(t):
    print(f"\n{'=' * 80}\n{t}\n{'=' * 80}")


def split(s):
    return s.str.split(";")


# 1 ---------------------------------------------------------------------------
h("1. Shapes / dtypes / missing / metaData")
for name, df in [("train", tr), ("test", te), ("sample_sub", ss), ("solvent_props", sp)]:
    print(f"\n--- {name}: {df.shape}")
    print(pd.DataFrame({"dtype": df.dtypes, "n_missing": df.isna().sum(), "n_unique": df.nunique()}))
print("\n--- metaData.csv (full)")
for _, r in meta.iterrows():
    print(f"\n[{r['column']}] file={r['file']} dtype={r['dtype']} units={r['units']}\n  {r['description']}")
print("\nsample_submission ids == test ids (same order):", ss["id"].equals(te["id"]))
print("train/test id overlap:", len(set(tr.id) & set(te.id)))

for df in (tr, te):
    df["solv_list"] = split(df["solvents"])
    df["combo"] = df["solv_list"].apply(lambda l: "+".join(sorted(l)))
    df["n_solv"] = df["solv_list"].str.len()

# 2 ---------------------------------------------------------------------------
h("2. frac_basis and conc_unit counts")
for col in ["frac_basis", "conc_unit"]:
    print(pd.concat({"train": tr[col].value_counts(), "test": te[col].value_counts()}, axis=1).fillna(0).astype(int))
print("\nfrac_basis x conc_unit (train):\n", pd.crosstab(tr.frac_basis, tr.conc_unit))
print("\nfrac_basis x conc_unit (test):\n", pd.crosstab(te.frac_basis, te.conc_unit))
# sanity: fractions sum to 1, conc_molL_est formula
for name, df in [("train", tr), ("test", te)]:
    s1 = split(df.solvent_fracs).apply(lambda l: sum(map(float, l)))
    s2 = split(df.solvent_fracs_mol).apply(lambda l: sum(map(float, l)))
    est = np.where(df.conc_unit == "mol/kg", df.conc_native * df.mixture_density_g_cm3, df.conc_native)
    print(f"{name}: frac sum range [{s1.min():.6f},{s1.max():.6f}], mol-frac sum range [{s2.min():.6f},{s2.max():.6f}], "
          f"max |conc_molL_est - formula| = {np.abs(est - df.conc_molL_est).max():.2e}")
    same = (df.frac_basis == "mol") & (df.solvent_fracs != df.solvent_fracs_mol)
    print(f"  rows with basis=mol but fracs != fracs_mol: {same.sum()}")

# 3 ---------------------------------------------------------------------------
h("3. Salts train vs test")
sc = pd.concat({"train": tr.salt_name.value_counts(), "test": te.salt_name.value_counts()}, axis=1).fillna(0).astype(int)
print(sc.sort_values("test", ascending=False))
print("test salts unseen in train:", sorted(set(te.salt_name) - set(tr.salt_name)) or "none")

# 4 ---------------------------------------------------------------------------
h("4. Solvent usage")
cnt_tr = Counter(s for l in tr.solv_list for s in l)
cnt_te = Counter(s for l in te.solv_list for s in l)
su = pd.DataFrame({"train_rows": pd.Series(cnt_tr), "test_rows": pd.Series(cnt_te)}).fillna(0).astype(int)
print(su.sort_values(["test_rows", "train_rows"], ascending=False).to_string())
print("\nrows with DEC: train", tr.solvents.str.contains(r"(^|;)DEC(;|$)").sum(),
      "test", te.solvents.str.contains(r"(^|;)DEC(;|$)").sum(), "of", len(te))
print("test solvents unseen in train:", sorted(set(cnt_te) - set(cnt_tr)))
print("\nn solvents per row:\n", pd.concat({"train": tr.n_solv.value_counts(), "test": te.n_solv.value_counts()}, axis=1).fillna(0).astype(int))

print("\nTop test combos (+ salts):")
tc = te.groupby("combo").agg(rows=("id", "size"), salts=("salt_name", lambda s: dict(Counter(s).most_common(4))))
tc = tc.sort_values("rows", ascending=False)
print(tc.head(25).to_string())

# analogue coverage: replace DEC with DMC / EMC and look for matching combo(+salt) in train
print("\nAnalogue coverage: test combo with DEC swapped for DMC/EMC (or DEC simply dropped) -> train rows")
tr_combo = tr.combo.value_counts()
tr_combo_salt = tr.groupby(["combo", "salt_name"]).size()
rows = []
for combo, g in te.groupby("combo"):
    parts = combo.split("+")
    partners = [p for p in parts if p != "DEC"]
    out = {"test_combo": combo, "test_rows": len(g)}
    for sub in ["DMC", "EMC", "<drop>"]:
        new = partners + ([sub] if sub != "<drop>" else [])
        key = "+".join(sorted(set(new))) if new else ""
        out[f"train_rows[{sub}]"] = int(tr_combo.get(key, 0))
        out[f"salt-matched[{sub}]"] = int(sum(tr_combo_salt.get((key, s), 0) for s in g.salt_name.unique()))
    rows.append(out)
ac = pd.DataFrame(rows).sort_values("test_rows", ascending=False)
print(ac.head(25).to_string(index=False))
anyhit = ac[[c for c in ac if c.startswith("train_rows")]].sum(axis=1) > 0
print(f"test rows whose combo has ANY analogue in train: {ac.loc[anyhit, 'test_rows'].sum()} / {len(te)}")
# row-level: does the same (partners, salt) appear in train at all?
te["partners"] = te.solv_list.apply(lambda l: "+".join(sorted(x for x in l if x != "DEC")))
tr_sets = tr.solv_list.apply(set)
partner_hit = te.partners.map(lambda p: bool(p) and tr_sets.apply(lambda s: set(p.split("+")) <= s).any())
print(f"test rows whose non-DEC partners all co-occur in some train row: {partner_hit.sum()} / {len(te)}")
print("pure DEC rows in test:", (te.combo == "DEC").sum())

# DEC fraction in test
dec_mol = te.apply(lambda r: float(dict(zip(r.solv_list, split(pd.Series([r.solvent_fracs_mol]))[0]))["DEC"]), axis=1)
te["dec_molfrac"] = dec_mol
print("\nDEC mole fraction in test:\n", dec_mol.describe())

# 5 ---------------------------------------------------------------------------
h("5. Distributions")
print("train k:\n", tr.k.describe())
print("train log_k:\n", tr.log_k.describe())
print("k<=0:", (tr.k <= 0).sum(), " | k<1e-3 mS/cm:", (tr.k < 1e-3).sum(), " | log_k NaN:", tr.log_k.isna().sum())
print("max |log10(k) - log_k|:", np.abs(np.log10(tr.k) - tr.log_k).max())
print("\nlowest log_k rows:\n", tr.nsmallest(8, "log_k")[["salt_name", "solvents", "conc_native", "conc_unit", "temperature_K", "log_k", "source_doi"]])
for c in ["temperature_K", "conc_native", "conc_molL_est", "mixture_density_g_cm3"]:
    print(f"\n{c}:\n", pd.concat({"train": tr[c].describe(), "test": te[c].describe()}, axis=1))
print("\nsubset:", tr.subset.value_counts().to_dict())
print("distinct T values: train", tr.temperature_K.round(2).nunique(), "test", te.temperature_K.round(2).nunique())

fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
ax[0].hist(tr.k, bins=80); ax[0].set_title("train k (mS/cm)")
ax[1].hist(tr.log_k, bins=80); ax[1].set_yscale("log"); ax[1].set_title("train log_k (log y)")
ax[2].hist([tr.temperature_K, te.temperature_K], bins=40, label=["train", "test"], density=True); ax[2].legend(); ax[2].set_title("T (K)")
ax[3].hist([tr.conc_molL_est, te.conc_molL_est], bins=40, label=["train", "test"], density=True); ax[3].legend(); ax[3].set_title("c est (mol/L)")
fig.tight_layout(); fig.savefig(FIGS / "distributions.png", dpi=110); plt.close(fig)

# 6 ---------------------------------------------------------------------------
h("6. Plots")
fig, ax = plt.subplots(figsize=(8, 6))
top = tr.salt_name.value_counts().index
for s in top:
    g = tr[tr.salt_name == s]
    ax.scatter(1000 / g.temperature_K, g.log_k, s=5, alpha=0.5, label=f"{s} ({len(g)})")
ax.set_ylim(-4, 1.5); ax.set_xlabel("1000/T (1/K)"); ax.set_ylabel("log10 k"); ax.legend(fontsize=7, markerscale=3)
ax.set_title("train: log k vs 1000/T by salt (y clipped at -4)")
fig.tight_layout(); fig.savefig(FIGS / "logk_vs_invT_by_salt.png", dpi=110); plt.close(fig)

near = tr[(tr.temperature_K - 298.15).abs() < 3]
fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for a, combo in zip(axs, ["DMC+EC", "EC+EMC"]):
    g = near[near.combo == combo]
    for (salt, fr), gg in g.groupby(["salt_name", "solvent_fracs_mol"]):
        if len(gg) < 3:
            continue
        gg = gg.sort_values("conc_molL_est")
        a.plot(gg.conc_molL_est, gg.k, "o-", ms=3, label=f"{salt} x={fr[:14]}")
    a.set_title(f"{combo}, T≈298 K ({len(g)} rows)"); a.set_xlabel("c est (mol/L)"); a.legend(fontsize=6)
axs[0].set_ylabel("k (mS/cm)")
fig.tight_layout(); fig.savefig(FIGS / "k_vs_c_298K_EC_DMC_EMC.png", dpi=110); plt.close(fig)
print("EC+DMC rows near 298K:", (near.combo == "DMC+EC").sum(), " EC+EMC:", (near.combo == "EC+EMC").sum())
# peak c per series
for combo in ["DMC+EC", "EC+EMC"]:
    g = near[near.combo == combo]
    pk = g.groupby(["salt_name", "solvent_fracs_mol"]).apply(lambda x: x.loc[x.k.idxmax(), "conc_molL_est"] if len(x) >= 4 else np.nan).dropna()
    print(f"{combo}: peak c (mol/L) per series with >=4 pts: n={len(pk)}, median={pk.median():.2f}, range=[{pk.min():.2f},{pk.max():.2f}]")

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
axs[0].scatter(tr.conc_molL_est, tr.temperature_K, s=3, alpha=0.3, label="train")
axs[0].scatter(te.conc_molL_est, te.temperature_K, s=3, alpha=0.3, label="test")
axs[0].set_xlabel("c est (mol/L)"); axs[0].set_ylabel("T (K)"); axs[0].legend(markerscale=4)
sc.plot.barh(ax=axs[1], logx=True); axs[1].set_title("rows per salt")
axs[2].hist(te.dec_molfrac, bins=30); axs[2].set_title("test DEC mole fraction")
fig.tight_layout(); fig.savefig(FIGS / "test_coverage.png", dpi=110); plt.close(fig)
print("saved:", sorted(p.name for p in FIGS.glob("*.png")))

# 7 ---------------------------------------------------------------------------
h("7. solvent_properties.csv vs solvent names used in data")
names = set(sp.name)
used = set(cnt_tr) | set(cnt_te)
print("used in data but missing from solvent_properties:", sorted(used - names))
print("in solvent_properties but unused:", sorted(names - used))
# SMILES agreement
smap = {}
for df in (tr, te):
    for a, b in zip(df.solvents, df.solvent_smiles):
        for n, s in zip(a.split(";"), b.split(";")):
            smap.setdefault(n, set()).add(s)
mism = {n: (s, sp.set_index("name").smiles.get(n)) for n, s in smap.items() if n in names and s != {sp.set_index("name").smiles[n]}}
print("SMILES mismatches data vs props:", mism or "none")
print("columns available in solvent_properties:", list(sp.columns))

# 8 ---------------------------------------------------------------------------
h("8. DOI overlap")
print("source_doi in test.csv:", "source_doi" in te.columns)
print("train unique DOIs:", tr.source_doi.nunique())
print("rows per conc_unit per DOI consistent:", (tr.groupby("source_doi").conc_unit.nunique() == 1).all())
print("train DOIs by #rows (top 10):\n", tr.source_doi.value_counts().head(10))
