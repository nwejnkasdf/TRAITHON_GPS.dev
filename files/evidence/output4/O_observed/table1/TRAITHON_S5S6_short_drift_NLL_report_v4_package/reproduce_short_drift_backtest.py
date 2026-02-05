# Reproduce: short-term drift backtest summaries (Adapter ON)
# Inputs:
#   - drift_thresholds_M50_adapterON_exact.csv
#   - weekly_nll_by_section_code.csv
#   - monthly_stats_2021_2022.csv
#
# Output:
#   - Summary tables (weekly/monthly)
#   - Q99 crossing bar charts

import os
import pandas as pd
import matplotlib.pyplot as plt

BASE = r"/mnt/data"
thr_path = os.path.join(BASE, "drift_thresholds_M50_adapterON_exact.csv")
weekly_path = os.path.join(BASE, "weekly_nll_by_section_code.csv")
monthly_path = os.path.join(BASE, "monthly_stats_2021_2022.csv")

thr = pd.read_csv(thr_path)
weekly = pd.read_csv(weekly_path)
monthly = pd.read_csv(monthly_path)

weekly["week_start"] = pd.to_datetime(weekly["week_start"], errors="coerce")
weekly["year"] = weekly["week_start"].dt.year
weekly["Count"] = weekly["Count"].astype(int)
monthly["month"] = pd.to_datetime(monthly["month"], errors="coerce")
monthly["Count"] = monthly["Count"].astype(int)

w = weekly[weekly["year"].between(2021, 2022)].merge(thr[["section","T_mean_nll","k_pre","p_tail"]], on="section", how="left")
m = monthly.merge(thr[["section","T_mean_nll","k_pre","p_tail"]], on="section", how="left")

for df in (w, m):
    for qcol, flag in [("Q99","ge_T_Q99"),("Q95","ge_T_Q95"),("Q90","ge_T_Q90"),("Median","ge_T_Median")]:
        df[flag] = (df[qcol] >= df["T_mean_nll"])

sec_week_summary = (w.groupby("section")
                    .agg(n_weeks=("week_start","count"),
                         median_count=("Count","median"),
                         p25_count=("Count", lambda x: x.quantile(0.25)),
                         p75_count=("Count", lambda x: x.quantile(0.75)),
                         weeks_Q99_ge_T=("ge_T_Q99","sum"))
                    .reset_index()
                    .sort_values("section"))
sec_week_summary["frac_weeks_Q99_ge_T"] = sec_week_summary["weeks_Q99_ge_T"] / sec_week_summary["n_weeks"]

sec_month_summary = (m.groupby("section")
                     .agg(n_months=("month","count"),
                          median_count=("Count","median"),
                          months_Q99_ge_T=("ge_T_Q99","sum"))
                     .reset_index()
                     .sort_values("section"))
sec_month_summary["frac_months_Q99_ge_T"] = sec_month_summary["months_Q99_ge_T"] / sec_month_summary["n_months"]

out_dir = os.path.join(BASE, "repro_outputs")
os.makedirs(out_dir, exist_ok=True)

sec_week_summary.to_csv(os.path.join(out_dir, "sec_week_summary.csv"), index=False, encoding="utf-8-sig")
sec_month_summary.to_csv(os.path.join(out_dir, "sec_month_summary.csv"), index=False, encoding="utf-8-sig")

# Weekly chart
plt.figure(figsize=(7,4))
plt.bar(sec_week_summary["section"], sec_week_summary["frac_weeks_Q99_ge_T"])
plt.xlabel("section")
plt.ylabel("Fraction of weeks with Q99 >= T")
plt.title("2021-2022: Weekly Q99 crossing fraction vs threshold T")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "weekly_q99_cross_fraction.png"), dpi=180)
plt.close()

# Monthly chart
plt.figure(figsize=(7,4))
plt.bar(sec_month_summary["section"], sec_month_summary["frac_months_Q99_ge_T"])
plt.xlabel("section")
plt.ylabel("Fraction of months with Q99 >= T")
plt.title("2021-2022: Monthly Q99 crossing fraction vs threshold T")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "monthly_q99_cross_fraction.png"), dpi=180)
plt.close()

print("Saved outputs to:", out_dir)
