"""Таблица метрик с 95% доверительными интервалами (замечание экспертов 16.10).

    python make_metrics_table.py

Читает out/metrics_final.json (результат evaluate_final.py) и строит таблицу для презентации:
метрики числом с 95% ДИ, как просили организаторы. Пишет out/metrics_table.md и .csv.
"""

import csv
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

root = Path(__file__).parent
report = json.load(open(root / "out" / "metrics_final.json", encoding="utf-8"))


def ci(v, key):
    c = v.get("ci", {})
    c = c.get(key) if isinstance(c, dict) else c
    return f"[{c[0]:.2f}; {c[1]:.2f}]" if c else ""


rows = []
for name, v in report.items():
    if not isinstance(v, dict) or name == "режимы":  # режимы — отдельная таблица в README/REPORT
        continue
    if name.startswith("quality_prob"):
        rows.append([name, "", "", f"{v['auc']:.3f} {ci(v, 'auc')}", "", "", "", "",
                     f"{v['pr_auc']:.3f}" if "pr_auc" in v else "",
                     f"средняя вероятность {v['mean_prob']:.2f} при доле брака {v['rate']:.2f}"])
    else:
        rows.append([name, v.get("n", ""), v.get("positives", ""),
                     f"{v['auc']:.3f} {ci(v, 'auc')}", f"{v['f1']:.3f} {ci(v, 'f1')}",
                     f"{v['sens']:.2f}", f"{v['spec']:.2f}", f"{v['bacc']:.2f}",
                     f"{v['pr_auc']:.3f} {ci(v, 'pr_auc')}" if "pr_auc" in v else "", ""])
rows.append(["macro-F1 по типам нарушений", "", "", "", f"{report['macro-F1 по типам нарушений']:.3f}", "", "", "", "", ""])

header = ["критерий", "n", "нарушений", "AUC [95% ДИ]", "F1 [95% ДИ]", "чувств.", "специф.", "сбал. точность", "PR-AUC [95% ДИ]",
          "примечание"]

with open(root / "out" / "metrics_table.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(header)
    w.writerows(rows)

md = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
text = "\n".join(md)
open(root / "out" / "metrics_table.md", "w", encoding="utf-8").write(text + "\n")
print(text)
print("\n→ out/metrics_table.md, out/metrics_table.csv")
