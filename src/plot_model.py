"""Charts for the model results (reads reports/ written by model.py)."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import eda  # reuse the chart style and colours

ROOT = Path(__file__).resolve().parent.parent
c = pd.read_csv(ROOT / "reports" / "calibration_test2025.csv")
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.bar(c.decile, c.actual * 100, color=eda.BLUE, width=0.62)
for x, v in zip(c.decile, c.actual * 100):
    ax.text(x, v + 0.04, f"{v:.1f}%", ha="center", color=eda.INK2, fontsize=9)
ax.set(xticks=range(1, 11), xlabel="Model risk group (1 = safest 10% of station-hours, 10 = riskiest 10%)",
       ylabel="Real delay rate in 2025 (%)", ylim=(0, 2.4))
ax.grid(axis="x", visible=False)
ax.set_title("The riskiest hours had 5x the delays of the safest (unseen 2025 data)")
fig.tight_layout(); fig.savefig(ROOT / "figures" / "06_model_risk_groups.png", dpi=160); plt.close(fig)

r = pd.read_csv(ROOT / "reports" / "model_results_test2025.csv", index_col=0)
fig, ax = plt.subplots(figsize=(8, 3.2))
s = r["auc"].iloc[::-1]
ax.barh(s.index, s.values, color=eda.BLUE, height=0.58)
for y, v in enumerate(s.values):
    ax.text(v + 0.004, y, f"{v:.3f}", va="center", color=eda.INK2, fontsize=9)
ax.set(xlim=(0.45, 0.7), xlabel="AUC on 2025 (0.5 = coin flip)")
ax.grid(axis="y", visible=False)
ax.set_title("Each model against simpler baselines")
fig.tight_layout(); fig.savefig(ROOT / "figures" / "07_model_auc.png", dpi=160); plt.close(fig)
print("ok")
