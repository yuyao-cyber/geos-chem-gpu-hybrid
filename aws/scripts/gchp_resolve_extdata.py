#!/usr/bin/env python3
"""Resolve every file a GCHP run directory's ExtData.rc can read for a date window,
against s3://geos-chem, and write the list of keys (with sizes) that are not yet on disk.

usage: gchp_resolve_extdata.py ExtData.rc START(YYYYMMDD) END(YYYYMMDD) OUT_PREFIX
Token handling: %y4 %m2 %d2 %h2 %n2.  Window: days START-1 .. END (+1 day),
months START-1 .. END+1, years START-1 .. END+1.  If no file matches inside the window
(dataset does not cover the year), the nearest available year is used, mirroring
Ext_AllowExtrap.  Static (token-free) files are taken as-is.
"""
import sys, re, json, subprocess, datetime as dt, os, collections

MAP = {"./MetDir/": "GEOS_0.25x0.3125/GEOS_FP/", "./ChemDir/": "CHEM_INPUTS/", "./HcoDir/": "HEMCO/"}
LOCAL = "/shared/ExtData/"
BUCKET = "geos-chem"

rc, s0, s1, out = sys.argv[1:5]
d0 = dt.date(int(s0[:4]), int(s0[4:6]), int(s0[6:])) - dt.timedelta(days=1)
d1 = dt.date(int(s1[:4]), int(s1[4:6]), int(s1[6:])) + dt.timedelta(days=1)
days = [d0 + dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]
def mshift(y, m, k):
    m += k; y += (m - 1) // 12; m = (m - 1) % 12 + 1; return y, m
months = set()
for d in days:
    for k in (-1, 0, 1): months.add(mshift(d.year, d.month, k))
years = sorted({y for y, _ in months})
target_year = days[len(days) // 2].year

_cache = {}
def listing(prefix):
    """All (key, size) under prefix (recursive)."""
    if prefix in _cache: return _cache[prefix]
    r = subprocess.run(["aws", "s3api", "list-objects-v2", "--bucket", BUCKET, "--prefix", prefix,
                        "--output", "json", "--query", "Contents[].[Key,Size]"],
                       capture_output=True, text=True)
    res = {}
    if r.returncode == 0 and r.stdout.strip() not in ("", "null"):
        for k, s in json.loads(r.stdout): res[k] = s
    _cache[prefix] = res
    return res

def tmpl_regex(t):
    rx = re.escape(t)
    for tok, g in (("%y4", r"(?P<y>\d{4})"), ("%m2", r"(?P<m>\d{2})"), ("%d2", r"(?P<d>\d{2})"),
                   ("%h2", r"(?P<h>\d{2})"), ("%n2", r"(?P<n>\d{2})")):
        esc = re.escape(tok)
        first = True
        while esc in rx:
            rx = rx.replace(esc, g if first else "(?P=%s)" % g[4], 1); first = False
    return re.compile("^" + rx + "$")

entries = []; clim = set()
in_prim = False
for line in open(rc):
    s = line.strip()
    if s.startswith("PrimaryExports%%"): in_prim = True; continue
    if in_prim and s.startswith("%%"): in_prim = False; continue
    if not in_prim or not s or s.startswith("#"): continue
    tok = s.split()
    f = tok[-1] if "/" in tok[-1] else tok[-2]
    if f.startswith("/dev/null"): continue
    for a, b in MAP.items():
        if f.startswith(a):
            entries.append(b + f[len(a):])
            if len(tok) > 2 and tok[2] == "Y": clim.add(b + f[len(a):])
            break
    else:
        print("UNMAPPED", f, file=sys.stderr)
templates = sorted(set(entries))

need = {}; report = collections.Counter(); unresolved = []
for t in templates:
    if "%" not in t:
        static_dir = t.rsplit("/", 1)[0] + "/"
        L = listing(static_dir)
        if t in L: need[t] = L[t]; report["static"] += 1
        else: unresolved.append(t)
        continue
    static = t[:t.index("%")]
    static = static.rsplit("/", 1)[0] + "/" if "/" in static else ""
    rx = tmpl_regex(t)
    L = listing(static)
    matches = []
    for k, sz in L.items():
        m = rx.match(k)
        if m: matches.append((m.groupdict(), k, sz))
    if not matches: unresolved.append(t); continue
    if t in clim and "%d2" not in t:
        # climatology: MAPL scans the whole year (it opens January first) -> all months of the
        # year nearest the simulation (or all files when the template has no year token)
        ys = sorted({int(g["y"]) for g, _, _ in matches if "y" in g})
        ny = min(ys, key=lambda y: (abs(y - target_year), -y)) if ys else None
        for g, k, sz in matches:
            if ny is None or int(g["y"]) == ny: need[k] = sz
        report["clim_year"] += 1
        continue
    def inwin(g, yshift=0):
        y = int(g["y"]) + yshift if "y" in g else None
        if "d" in g and y is not None:
            try: d = dt.date(y, int(g["m"]), int(g["d"]))
            except ValueError: return False
            return d0 <= d <= d1
        if "m" in g and y is not None: return (y, int(g["m"])) in months
        if "m" in g: return any(int(g["m"]) == mm for _, mm in months)
        if y is not None: return y in years
        return True
    sel = [(k, sz) for g, k, sz in matches if inwin(g)]
    if sel: report["window"] += 1
    else:
        ys = sorted({int(g["y"]) for g, _, _ in matches if "y" in g})
        if ys:
            ny = min(ys, key=lambda y: (abs(y - target_year), -y))
            sel = [(k, sz) for g, k, sz in matches if "y" in g and int(g["y"]) == ny and inwin(g, target_year - ny)]
            if not sel:   # e.g. annual file only in that year
                sel = [(k, sz) for g, k, sz in matches if "y" in g and int(g["y"]) == ny][:13]
            report["nearest_year"] += 1
            print("NEAREST %d  %s" % (ny, t), file=sys.stderr)
        if not sel: unresolved.append(t); continue
    for k, sz in sel: need[k] = sz

missing = {k: s for k, s in need.items() if not os.path.exists(LOCAL + k)}
with open(out + "_needed.txt", "w") as fh:
    for k in sorted(need): fh.write("%s %d\n" % (k, need[k]))
with open(out + "_missing.txt", "w") as fh:
    for k in sorted(missing): fh.write(k + "\n")
with open(out + "_unresolved.txt", "w") as fh:
    for t in unresolved: fh.write(t + "\n")
GB = 1e9
print("templates=%d resolved=%s unresolved=%d" % (len(templates), dict(report), len(unresolved)))
print("needed files=%d  %.1f GB ; missing locally=%d  %.1f GB" %
      (len(need), sum(need.values()) / GB, len(missing), sum(missing.values()) / GB))
top = collections.Counter()
for k, s in missing.items(): top["/".join(k.split("/")[:3])] += s
for p, s in top.most_common(15): print("  %8.1f GB  %s" % (s / GB, p))
