import itertools, re, subprocess, sys, json, os
base = open('dev/candidates/rollout.py').read()
grid = {
 'PARK_SPREAD': [0, 2, 4, 8],
 'NODE_DELAY': [1, 2, 4],
 'YIELD': [True, False],
 'SOFT_EVENT': [0.12, 0.25],
}
sets = sys.argv[1:] or ['test_cases', 'dev/cases/dev']
res = []
for i, vals in enumerate(itertools.product(*grid.values())):
    src = base
    for k, v in zip(grid, vals):
        src = re.sub(r'^%s = .*$' % k, '%s = %r' % (k, v), src, count=1, flags=re.M)
    p = 'dev/sweep/r%d.py' % i
    open(p, 'w').write(src)
    out = subprocess.run([sys.executable, 'dev/bench.py', '--routing', p, '--jobs', '7', '--quiet'] + sets,
                         capture_output=True, text=True).stdout
    m = re.search(r'^ALL\s+\d+\s+([\d.]+)', out, re.M)
    score = float(m.group(1)) if m else -1
    res.append((score, dict(zip(grid, vals)), p))
    print('%.3f %s' % (score, dict(zip(grid, vals))), flush=True)
res.sort(key=lambda r: -r[0])
print('TOP'); [print('%.3f %s %s' % r) for r in res[:5]]
