# This script generates the Critical Difference (CD) diagram to show the
#   results of the Nemenyi post hoc test for the statistical differences
#   between the APFDc of Different ML ranking algorithms.
# Due to the PyQt5 conflict between Understand and Orange,
#   this file should be executed separately after the results command is executed.

import pandas as pd
from scipy.stats import rankdata
import Orange
import matplotlib.pyplot as plt
import sys
from pathlib import Path

results_path = Path(sys.argv[1])
m_df = pd.read_csv(results_path / "RQ2" / "rq2_ranker_res.csv")
res_rank = {}
for _, r in m_df.iterrows():
    f1_l = [f1 for alg, f1 in r.items()]
    f1_ranks = rankdata([-1 * f1 for f1 in f1_l])
    i = 0
    for alg, f1 in r.items():
        res_rank.setdefault(alg, []).append(f1_ranks[i])
        i += 1
res_rank = pd.DataFrame(res_rank)

names = res_rank.columns.tolist()
avranks = [avr for alg, avr in res_rank.mean().items()]
cd = Orange.evaluation.compute_CD(avranks, len(res_rank), alpha="0.05", test="nemenyi")
Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
plt.savefig(
    str(results_path / "RQ2" / "rq2_ranker_nemeyni_{:.2f}.png".format(cd)),
    bbox_inches="tight",
    facecolor="white",
)
