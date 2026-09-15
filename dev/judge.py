#!/usr/bin/env python3
"""
Compare candidate routing files head-to-head on the same case sets.

Usage:
    python3 dev/judge.py dev/candidates/*.py dev/baselines/greedy_sp.py --sets test_cases dev/cases/holdout
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", nargs="+")
    ap.add_argument("--sets", nargs="+", default=["test_cases", "dev/cases/holdout"])
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()

    table = {}
    percase = {}
    for cand in args.candidates:
        name = os.path.splitext(os.path.basename(cand))[0]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            out = tf.name
        subprocess.run([sys.executable, os.path.join(REPO, "dev/bench.py"), "--routing", cand,
                        "--jobs", str(args.jobs), "--quiet", "--json", out] + args.sets,
                       cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(out) as f:
            res = json.load(f)
        os.unlink(out)
        groups = {}
        for r in res:
            parts = r["case"].split("/")
            g = ("official/" if "test_cases" in r["case"] else "") + parts[-2]
            groups.setdefault(g, []).append(r)
            percase.setdefault(r["case"], {})[name] = r["score"]
        row = {g: sum(x["score"] for x in rs) / len(rs) for g, rs in groups.items()}
        row["ALL"] = sum(x["score"] for x in res) / len(res)
        row["_deliv"] = 100.0 * sum(x["delivered"] for x in res) / max(1, sum(x["total"] for x in res))
        row["_maxcall"] = max(x["max_call"] for x in res)
        row["_maxwall"] = max(x["wall"] for x in res)
        row["_exc"] = sum(x["exceptions"] + x["timeouts"] for x in res)
        table[name] = row

    cols = sorted({k for row in table.values() for k in row if not k.startswith("_") and k != "ALL"}) + ["ALL"]
    print("%-14s" % "candidate" + "".join("%11s" % c[-11:] for c in cols) + "%8s%9s%9s%5s" % ("deliv%", "maxcall", "maxwall", "exc"))
    for name, row in sorted(table.items(), key=lambda kv: -kv[1]["ALL"]):
        print("%-14s" % name[:14] + "".join("%11.2f" % row.get(c, float("nan")) for c in cols)
              + "%7.1f%%%8.3fs%8.2fs%5d" % (row["_deliv"], row["_maxcall"], row["_maxwall"], row["_exc"]))

    # oracle: best candidate per case, shows headroom from combining ideas
    names = list(table)
    best = sum(max(v.values()) for v in percase.values()) / len(percase)
    print("\nper-case oracle (max over candidates): %.2f" % best)
    wins = {n: 0 for n in names}
    for v in percase.values():
        top = max(v.values())
        for n in names:
            if v.get(n, -1) >= top - 1e-9:
                wins[n] += 1
    print("per-case wins (ties count for all): " + ", ".join("%s=%d" % kv for kv in sorted(wins.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
