#!/usr/bin/env bash
# iTwin-Forest on the five scenarios x three training seeds (one run after the other; adapt to your scheduler).
set -e
DATA=${1:-datasets}
for sc in random co ew we site; do
  for seed in 2025 2024 2023; do
    python -m itwin_forest.train --data_root "$DATA" --scenario "$sc" --seed "$seed" --tag itwin_forest
  done
done
