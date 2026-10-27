"""Mean +- sample std over seeds of runs/<scenario>_<tag>_s*/metrics.json.

    python scripts/aggregate.py runs
"""
import glob
import json
import re
import sys
from collections import defaultdict

import numpy as np

groups = defaultdict(list)
for f in glob.glob(f"{sys.argv[1]}/*/metrics.json"):
    groups[re.sub(r"_s\d+$", "", f.replace("\\", "/").split("/")[-2])].append(json.load(open(f))["metrics"])
for g, ms in sorted(groups.items()):
    row = []
    for cls in ("dead", "all"):
        for m in ("f1", "precision", "recall", "iou"):
            v = 100 * np.array([x[cls][m] for x in ms])
            row.append(f"{cls}-{m} {v.mean():.2f}±{v.std(ddof=1) if len(v) > 1 else 0:.2f}")
    print(f"{g} (n={len(ms)}): " + "  ".join(row))
