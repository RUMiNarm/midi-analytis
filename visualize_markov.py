#!/usr/bin/env python3
"""
visualize_markov.py
-------------------
json 化された 2 階マルコフ遷移確率をヒートマップで表示・保存します。

$ python visualize_markov.py models/pitch_model.json pitch_heatmap.png
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_matrix(json_path: Path) -> pd.DataFrame:
    data = json.loads(json_path.read_text())
    records = []
    for state, next_dict in data.items():
        for nxt, prob in next_dict.items():
            records.append({"state": state, "next": nxt, "prob": prob})
    df = pd.DataFrame(records)
    return df.pivot(index="state", columns="next", values="prob").fillna(0)


def main():
    if len(sys.argv) < 3:
        print("usage: visualize_markov.py model.json out.png")
        sys.exit(1)
    json_path, out_png = Path(sys.argv[1]), Path(sys.argv[2])
    mat = load_matrix(json_path)

    plt.figure(figsize=(10, 8))
    plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.colorbar(label="transition prob.")
    plt.xticks(range(len(mat.columns)), mat.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(mat.index)), mat.index, fontsize=6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"ヒートマップを保存しました → {out_png}")


if __name__ == "__main__":
    main()
