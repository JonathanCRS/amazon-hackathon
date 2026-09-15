#!/usr/bin/env python3
"""
Benchmark a routing module against many test cases using the real GameEngine.

Usage:
    python3 dev/bench.py --routing ar_hackathon/api/routing.py dev/cases/level1 dev/cases/level2 ...
    python3 dev/bench.py --routing path/to/candidate.py test_cases --jobs 4 --json out.json

Directories are searched recursively for *.json (schema.json is skipped).

Modes:
    default          each case runs with a freshly imported copy of the module,
                     calls are timed directly (a call > 1s is treated as a wait,
                     exactly like the engine's timeout) - fast.
    --strict         use the engine's real thread-pool safe_execute.
    --shared-module  import the module ONCE and run every case sequentially in
                     one process, like a grader that never reloads your file.
                     Catches module-level state leaking between games.
"""

import argparse
import glob
import heapq
import importlib.util
import json
import math
import os
import sys
import time
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import logging
logging.disable(logging.WARNING)


def find_cases(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(glob.glob(os.path.join(p, "**", "*.json"), recursive=True)):
                if os.path.basename(f) != "schema.json":
                    out.append(f)
        elif p.endswith(".json"):
            out.append(p)
    return out


def load_module(path, tag):
    name = "cand_%s_%d" % (tag, os.getpid())
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def upper_bound(case_path):
    """Optimistic score: every pod carried along its shortest path the instant it spawns."""
    with open(case_path) as f:
        d = json.load(f)
    adj = {}
    for e in d["edges"]:
        w = int(math.ceil(e["weight"]))
        adj.setdefault(e["from_node"], []).append((e["to_node"], w))
        if e.get("bidirectional", True):
            adj.setdefault(e["to_node"], []).append((e["from_node"], w))
    total = 0.0
    for p in d["pods"]:
        s, t = p["source_node"], p["destination_station"]
        dist = {s: 0}
        pq = [(0, s)]
        while pq:
            du, u = heapq.heappop(pq)
            if u == t:
                break
            if du > dist[u]:
                continue
            for v, w in adj.get(u, []):
                if du + w < dist.get(v, 1e18):
                    dist[v] = du + w
                    heapq.heappush(pq, (du + w, v))
        W = dist.get(t)
        if W is not None:
            total += math.exp(-max(0, W - 1) / 50.0)
    return 100.0 * total / max(1, len(d["pods"]))


class CallStats:
    def __init__(self):
        self.calls = 0
        self.max_call = 0.0
        self.total = 0.0
        self.exceptions = 0
        self.timeouts = 0
        self.invalid = 0
        self.last_exc = None


def install_fast_executor(stats, strict):
    import ar_hackathon.engine.game_engine as ge
    if not hasattr(ge, "_orig_safe_execute"):
        ge._orig_safe_execute = ge.safe_execute
    orig = ge._orig_safe_execute

    def timed(func, *args, timeout_seconds=10, default_return_value=None, **kwargs):
        t0 = time.perf_counter()
        if strict:
            r = orig(func, *args, timeout_seconds=timeout_seconds,
                     default_return_value=default_return_value, **kwargs)
        else:
            try:
                r = func(*args, **kwargs)
            except Exception as e:  # noqa
                stats.exceptions += 1
                stats.last_exc = "%s: %s" % (type(e).__name__, e)
                r = default_return_value
        dt = time.perf_counter() - t0
        stats.calls += 1
        stats.total += dt
        stats.max_call = max(stats.max_call, dt)
        if dt > timeout_seconds:
            stats.timeouts += 1
            r = default_return_value
        return r

    ge.safe_execute = timed


def run_case(mod, case_path, strict=False):
    from ar_hackathon.engine.game_engine import GameEngine
    from ar_hackathon.utils import routing_utils
    stats = CallStats()
    install_fast_executor(stats, strict)
    fn = mod.drive_unit_next_move

    t0 = time.perf_counter()
    engine = GameEngine(case_path, fn)
    # count invalid (non-None, rejected) moves for diagnostics
    import ar_hackathon.engine.game_engine as ge
    orig_valid = routing_utils.is_valid_move

    def counting_valid(gs, unit, nxt):
        ok = orig_valid(gs, unit, nxt)
        if not ok and nxt is not None and nxt != unit.current_node:
            stats.invalid += 1
        return ok

    ge.is_valid_move = counting_valid
    res = engine.run_until_finished()
    ge.is_valid_move = orig_valid
    wall = time.perf_counter() - t0
    return {
        "case": case_path,
        "score": res["score"],
        "delivered": res["delivered_pods"],
        "total": res["total_pods"],
        "steps": res["total_time_steps"],
        "wall": wall,
        "max_call": stats.max_call,
        "calls": stats.calls,
        "exceptions": stats.exceptions,
        "timeouts": stats.timeouts,
        "invalid": stats.invalid,
        "last_exc": stats.last_exc,
        "ub": upper_bound(case_path),
    }


def _worker(job):
    routing, case_path, strict = job
    try:
        mod = load_module(routing, "w")
        return run_case(mod, case_path, strict)
    except Exception as e:
        import traceback
        return {"case": case_path, "score": 0.0, "delivered": 0, "total": 0, "steps": 0,
                "wall": 0, "max_call": 0, "calls": 0, "exceptions": 1, "timeouts": 0, "invalid": 0,
                "last_exc": "HARNESS: " + traceback.format_exc()[-500:], "ub": 0}


def group_of(case_path):
    parts = case_path.replace("\\", "/").split("/")
    return parts[-2] if len(parts) >= 2 else "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--routing", default=os.path.join(REPO, "ar_hackathon/api/routing.py"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--shared-module", action="store_true")
    ap.add_argument("--json", help="write per-case results here")
    ap.add_argument("--quiet", action="store_true", help="only print the summary")
    args = ap.parse_args()

    cases = find_cases(args.paths)
    if not cases:
        print("no cases found")
        return 1
    routing = os.path.abspath(args.routing)

    t0 = time.perf_counter()
    if args.shared_module:
        mod = load_module(routing, "shared")
        results = [run_case(mod, c, args.strict) for c in cases]
    else:
        jobs = [(routing, c, args.strict) for c in cases]
        with Pool(processes=max(1, args.jobs), maxtasksperchild=1) as pool:
            results = pool.map(_worker, jobs, chunksize=1)
    elapsed = time.perf_counter() - t0

    if not args.quiet:
        print("%-44s %7s %7s %6s %6s %7s %7s %4s %4s" % ("case", "score", "ub", "deliv", "steps", "wall", "maxcall", "exc", "inv"))
        for r in results:
            name = "/".join(r["case"].split("/")[-2:])
            print("%-44s %7.2f %7.2f %3d/%-3d %5d %6.2fs %6.3fs %4d %4d" % (
                name[-44:], r["score"], r["ub"], r["delivered"], r["total"], r["steps"],
                r["wall"], r["max_call"], r["exceptions"] + r["timeouts"], r["invalid"]))
            if r["last_exc"]:
                print("    last exception: %s" % r["last_exc"][:300])

    groups = {}
    for r in results:
        groups.setdefault(group_of(r["case"]), []).append(r)
    print("\n%-12s %5s %9s %9s %8s %8s %9s %6s" % ("group", "n", "mean", "ub_mean", "deliv%", "maxwall", "maxcall", "exc"))
    for g in sorted(groups):
        rs = groups[g]
        dl = sum(r["delivered"] for r in rs)
        tot = sum(r["total"] for r in rs)
        print("%-12s %5d %9.3f %9.3f %7.1f%% %7.2fs %8.3fs %6d" % (
            g, len(rs), sum(r["score"] for r in rs) / len(rs), sum(r["ub"] for r in rs) / len(rs),
            100.0 * dl / max(1, tot), max(r["wall"] for r in rs), max(r["max_call"] for r in rs),
            sum(r["exceptions"] + r["timeouts"] for r in rs)))
    n = len(results)
    print("%-12s %5d %9.3f %9.3f   total=%.2f  elapsed=%.1fs" % (
        "ALL", n, sum(r["score"] for r in results) / n, sum(r["ub"] for r in results) / n,
        sum(r["score"] for r in results), elapsed))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
