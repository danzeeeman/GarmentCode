"""Patchwork: a garment's pieces broken into smaller ones along lines laid over the garment, every new piece sewn to its neighbours.

This is a line-for-line port of libgarment's edit.cpp (cut_line, patchwork and the edit ops they use) in the AtelierCutroom app, so the
generator and the app make the same pieces for the same design. It works on the pattern dict ({"panels", "stitches"}) after assembly:

    pattern = piece.assembly()
    apply_patchwork(pattern.pattern, design)

The design's "patchwork" group:
    style       none | triangles | shards | strips
    piece_size  cm: the side of a triangle, the width of a strip, about the size of a shard
    angle       degrees: turns the strips (0: vertical) and the triangle grid
    seed        the shards' random lines (splitmix64, the same numbers as the C++ engine)
    parts       all | torso | skirt | sleeves
    side        both | left | right   (the wearer's left is +x)
    mirror      the lines reflected at the centre (left and right alike); off: they run on across the body

Panels are visited in name order, as the C++ engine (nlohmann::json objects are sorted) does.
"""
import copy
import math
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import svgpathtools as svgpath

MASK64 = 0xFFFFFFFFFFFFFFFF


class EditError(Exception):
    pass


# ----------------------------------------------------------------------------------------------- helpers (edit.cpp, anonymous namespace)

def _vtx(panel, i):
    v = panel['vertices'][i]
    return np.array([float(v[0]), float(v[1])])


def _c(p):
    return complex(p[0], p[1])


def _from_c(c):
    return np.array([c.real, c.imag])


def _rel_to_abs_2d(start, end, rel):
    edge = end - start
    perp = np.array([-edge[1], edge[0]])
    return start + rel[0] * edge + rel[1] * perp


def _abs_to_rel_2d(start, end, point):
    edge = end - start
    edge_len = np.linalg.norm(edge)
    vec = point - start
    projected_len = edge.dot(vec) / edge_len
    rx = projected_len / edge_len
    vert = vec - edge * rx
    ry = np.linalg.norm(vert) / edge_len
    cross = edge[0] * vec[1] - edge[1] * vec[0]
    ry *= np.sign(cross)
    return np.array([rx, ry])


def _edge_curve(panel, e):
    ej = panel['edges'][e]
    s = _vtx(panel, ej['endpoints'][0])
    en = _vtx(panel, ej['endpoints'][1])
    if 'curvature' not in ej or ej['curvature'] is None:
        return svgpath.Line(_c(s), _c(en))
    c = ej['curvature']
    t = c['type']
    p = c['params']
    rel = lambda q: _rel_to_abs_2d(s, en, np.array([float(q[0]), float(q[1])]))
    if t == 'quadratic':
        return svgpath.QuadraticBezier(_c(s), _c(rel(p[0])), _c(en))
    if t == 'cubic':
        return svgpath.CubicBezier(_c(s), _c(rel(p[0])), _c(rel(p[1])), _c(en))
    if t == 'circle':
        r = float(p[0])
        return svgpath.Arc(_c(s), complex(r, r), 0.0, bool(p[1]), bool(p[2]), _c(en))
    raise EditError('unknown curvature type: ' + t)


def _curvature_of(c, s, e):
    if isinstance(c, svgpath.QuadraticBezier):
        r = _abs_to_rel_2d(s, e, _from_c(c.control))
        return {'type': 'quadratic', 'params': [[float(r[0]), float(r[1])]]}
    if isinstance(c, svgpath.CubicBezier):
        r1 = _abs_to_rel_2d(s, e, _from_c(c.control1))
        r2 = _abs_to_rel_2d(s, e, _from_c(c.control2))
        return {'type': 'cubic', 'params': [[float(r1[0]), float(r1[1])], [float(r2[0]), float(r2[1])]]}
    if isinstance(c, svgpath.Arc):
        return {'type': 'circle', 'params': [float(c.radius.real), 1 if c.large_arc else 0, 1 if c.sweep else 0]}
    return None


def _sample_loop(panel, per_edge=12):
    pts = []
    for e in range(len(panel['edges'])):
        c = _edge_curve(panel, e)
        for k in range(per_edge):
            pts.append(_from_c(c.point(k / per_edge)))
    return pts


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _properly_cross(a, b, c, d):
    o1, o2, o3, o4 = _orient(a, b, c), _orient(a, b, d), _orient(c, d, a), _orient(c, d, b)
    return (((o1 > 1e-9 and o2 < -1e-9) or (o1 < -1e-9 and o2 > 1e-9))
            and ((o3 > 1e-9 and o4 < -1e-9) or (o3 < -1e-9 and o4 > 1e-9)))


def _rotation(panel):
    r = panel['rotation']
    return Rotation.from_euler('XYZ', [float(r[0]), float(r[1]), float(r[2])], degrees=True)


def _place(panel, p):
    t = panel['translation']
    return _rotation(panel).apply([p[0], p[1], 0.0]) + np.array([float(t[0]), float(t[1]), float(t[2])])


def _vindex(panel, e, side):
    return int(panel['edges'][e]['endpoints'][side])


def _refs_to(pattern, panel, edge):
    out = []
    for i, st in enumerate(pattern['stitches']):
        for s in range(2):
            side = st[s]
            if isinstance(side, dict) and side.get('panel') == panel and side.get('edge', -1) == edge:
                out.append((i, s))
    return out


def _shift_stitch_edges(pattern, panel, from_edge, delta):
    for st in pattern['stitches']:
        for side in st:
            if isinstance(side, dict) and side.get('panel') == panel and side.get('edge', -1) >= from_edge:
                side['edge'] = side['edge'] + delta


def _split_edge_local(pattern, name, e, f):
    panel = pattern['panels'][name]
    c = _edge_curve(panel, e)
    length = c.length()
    t = f if isinstance(c, svgpath.Line) else c.ilength(f * length)
    first, second = c.split(t)
    mid = _from_c(first.end)
    nv = len(panel['vertices'])
    panel['vertices'].append([float(mid[0]), float(mid[1])])
    orig = panel['edges'][e]
    a, b = int(orig['endpoints'][0]), int(orig['endpoints'][1])
    pa, pb = _vtx(panel, a), _vtx(panel, b)
    e1, e2 = copy.deepcopy(orig), copy.deepcopy(orig)
    e1['endpoints'] = [a, nv]
    e2['endpoints'] = [nv, b]
    e1.pop('curvature', None)
    e2.pop('curvature', None)
    c1, c2 = _curvature_of(first, pa, mid), _curvature_of(second, mid, pb)
    if c1 is not None:
        e1['curvature'] = c1
    if c2 is not None:
        e2['curvature'] = c2
    panel['edges'][e] = e1
    panel['edges'].insert(e + 1, e2)
    _shift_stitch_edges(pattern, name, e + 1, 1)
    return nv


# ----------------------------------------------------------------------------------------------- geometry / validity

def edge_length(panel, edge):
    return _edge_curve(panel, edge).length()


def panel_area(panel):
    pts = _sample_loop(panel, 32)
    n = len(pts)
    a = 0.0
    for i in range(n):
        a += pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
    return abs(a) / 2


def validate_panel(panel):
    """Empty string when the panel is valid (closed loop, indices in range, no self intersection, positive area)."""
    try:
        if 'vertices' not in panel or 'edges' not in panel:
            return 'panel has no vertices / edges'
        nv, ne = len(panel['vertices']), len(panel['edges'])
        if ne < 3:
            return 'a panel needs at least 3 edges'
        for i in range(ne):
            a, b = _vindex(panel, i, 0), _vindex(panel, i, 1)
            if a < 0 or b < 0 or a >= nv or b >= nv:
                return 'edge refers to a missing vertex'
            if a == b:
                return 'an edge has the same start and end'
            if b != _vindex(panel, (i + 1) % ne, 0):
                return 'edges do not form a closed loop'
            if np.linalg.norm(_vtx(panel, a) - _vtx(panel, b)) < 1e-4:
                return 'an edge has zero length'
        if panel_area(panel) < 0.5:
            return 'panel area is too small'
        pts = _sample_loop(panel)
        n = len(pts)
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                if _properly_cross(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n]):
                    return 'the outline crosses itself'
        return ''
    except Exception as ex:  # same as the C++ catch
        return 'invalid panel: ' + str(ex)


# ----------------------------------------------------------------------------------------------- ops

def add_point(pat, name, edge, t):
    panel = pat['panels'][name]
    if edge < 0 or edge >= len(panel['edges']):
        raise EditError('no such edge')
    if t < 0.02 or t > 0.98:
        raise EditError('too close to the end of the edge')
    partners = []  # (stitch, my_side, panel, edge, same_dir)
    for st_i, side in _refs_to(pat, name, edge):
        other = pat['stitches'][st_i][1 - side]
        if not isinstance(other, dict):
            continue
        qn = other.get('panel', '')
        if qn == name:
            raise EditError('this edge is sewn to its own panel: not supported')
        j = other.get('edge', -1)
        q = pat['panels'][qn]
        s0 = _place(panel, _vtx(panel, _vindex(panel, edge, 0)))
        s1 = _place(panel, _vtx(panel, _vindex(panel, edge, 1)))
        q0 = _place(q, _vtx(q, _vindex(q, j, 0)))
        q1 = _place(q, _vtx(q, _vindex(q, j, 1)))
        same_dir = np.linalg.norm(s0 - q0) + np.linalg.norm(s1 - q1) <= np.linalg.norm(s0 - q1) + np.linalg.norm(s1 - q0)
        partners.append((st_i, side, qn, j, same_dir))
    _split_edge_local(pat, name, edge, t)
    for st_i, my_side, qn, j, same_dir in partners:
        f = t if same_dir else 1.0 - t
        _split_edge_local(pat, qn, j, f)
        st = pat['stitches'][st_i]
        mine2, theirs2 = copy.deepcopy(st[my_side]), copy.deepcopy(st[1 - my_side])
        mine2['edge'] = edge + 1
        if not same_dir:
            st[1 - my_side]['edge'] = j + 1
            theirs2['edge'] = j
        else:
            theirs2['edge'] = j + 1
        pat['stitches'].append([mine2, theirs2] if my_side == 0 else [theirs2, mine2])
    touched = {name} | {p[2] for p in partners}
    for n in sorted(touched):
        err = validate_panel(pat['panels'][n])
        if err:
            raise EditError(err)


def cut(pat, name, va, vb):
    parent = pat['panels'][name]
    ne = len(parent['edges'])
    if va == vb or va < 0 or vb < 0 or va >= len(parent['vertices']) or vb >= len(parent['vertices']):
        raise EditError('choose two different points')
    ia = ib = -1
    for i in range(ne):
        if _vindex(parent, i, 0) == va:
            ia = i
        if _vindex(parent, i, 0) == vb:
            ib = i
    if ia < 0 or ib < 0:
        raise EditError('a point is not on the outline')
    n1, n2 = (ib - ia + ne) % ne, (ia - ib + ne) % ne
    if n1 < 2 or n2 < 2:
        raise EditError('the points are neighbours: a cut needs at least two edges on each side')
    name_a, name_b = name + '_a', name + '_b'
    k = 2
    while name_a in pat['panels'] or name_b in pat['panels']:
        name_a, name_b = name + '_a' + str(k), name + '_b' + str(k)
        k += 1
    edge_to_child = {}

    def build(first, count, from_vertex, to_vertex, child):
        out = {key: copy.deepcopy(v) for key, v in parent.items() if key not in ('vertices', 'edges')}
        remap = {}
        out['vertices'] = []
        out['edges'] = []

        def mapv(v):
            if v in remap:
                return remap[v]
            remap[v] = len(out['vertices'])
            out['vertices'].append(copy.deepcopy(parent['vertices'][v]))
            return remap[v]

        for kk in range(count):
            e = (first + kk) % ne
            ej = copy.deepcopy(parent['edges'][e])
            ej['endpoints'] = [mapv(_vindex(parent, e, 0)), mapv(_vindex(parent, e, 1))]
            edge_to_child[e] = (child, kk)
            out['edges'].append(ej)
        out['edges'].append({'endpoints': [mapv(to_vertex), mapv(from_vertex)], 'label': 'cut'})
        return out

    c1 = build(ia, n1, va, vb, 0)
    c2 = build(ib, n2, vb, va, 1)
    for c in (c1, c2):
        err = validate_panel(c)
        if err:
            raise EditError('the cut makes an invalid panel: ' + err)
    for st in pat['stitches']:
        for side in st:
            if isinstance(side, dict) and side.get('panel') == name:
                f = edge_to_child.get(side.get('edge', -1))
                if f is None:
                    raise EditError('a stitch refers to a missing edge')
                side['panel'] = name_a if f[0] == 0 else name_b
                side['edge'] = f[1]
    del pat['panels'][name]
    pat['panels'][name_a] = c1
    pat['panels'][name_b] = c2
    pat['stitches'].append([{'panel': name_a, 'edge': n1}, {'panel': name_b, 'edge': n2}])


# ----------------------------------------------------------------------------------------------- cut along a line, patchwork

def _line_crossings(panel, p0, p1):
    """Where the line through p0 and p1 crosses the outline: [(edge, fraction of its length, position along the line, point)].
    The outline is walked as one loop of samples, each on one side of the line or (within 1e-6 cm) on it; a crossing is where the outline
    really passes from one side to the other (a line touching a corner is none). The same as the C++ engine."""
    N = 64
    k_on = 1e-6
    out = []
    d = p1 - p0
    dl = np.linalg.norm(d)
    if dl < 1e-9:
        return out
    u = d / dl
    nrm = np.array([-u[1], u[0]])
    smp = []  # (edge, at, next, cum, seg, total, dist, side)
    for e in range(len(panel['edges'])):
        c = _edge_curve(panel, e)
        pts = [_from_c(c.point(k / N)) for k in range(N + 1)]
        cum = [0.0] * (N + 1)
        for k in range(1, N + 1):
            cum[k] = cum[k - 1] + np.linalg.norm(pts[k] - pts[k - 1])
        if cum[N] < 1e-9:
            continue
        for k in range(N):
            at, nx = pts[k], pts[k + 1]
            dist = (at - p0).dot(nrm)
            side = 1 if dist > k_on else (-1 if dist < -k_on else 0)
            smp.append((e, at, nx, cum[k], np.linalg.norm(nx - at), cum[N], dist, side))
    m = len(smp)
    start = next((i for i in range(m) if smp[i][7] != 0), m)
    if start == m:
        return out
    prev = start
    for step in range(1, m + 1):
        j = (start + step) % m
        if smp[j][7] == 0:
            continue
        if smp[j][7] != smp[prev][7]:
            after = (prev + 1) % m
            if after == j:
                e, at, nx, cumk, seg, total, dist, _ = smp[prev]
                dn = (nx - p0).dot(nrm)
                w = dist / (dist - dn)
                pt = at + (nx - at) * w
                out.append((e, (cumk + w * seg) / total, (pt - p0).dot(u), pt))
            else:
                e, at, _, cumk, _, total, _, _ = smp[after]
                out.append((e, cumk / total, (at - p0).dot(u), at))
        prev = j
    out.sort(key=lambda c: c[2])
    return out


def _vertex_at(pat, name, c, snap):
    panel = pat['panels'][name]
    edge, f = c[0], c[1]
    length = edge_length(panel, edge)
    if f * length < snap:
        return _vindex(panel, edge, 0)
    if (1 - f) * length < snap:
        return _vindex(panel, edge, 1)
    try:
        add_point(pat, name, edge, f)
    except EditError:
        return -1
    return len(pat['panels'][name]['vertices']) - 1


def _cut_between(pat, name, p0, p1, A, B, min_area):
    work = copy.deepcopy(pat)
    snap = 1.0
    va = _vertex_at(work, name, A, snap)
    if va < 0:
        raise EditError('cannot put a point where the line meets the outline')
    best, nb = 1e9, None
    for c in _line_crossings(work['panels'][name], p0, p1):
        dist = np.linalg.norm(c[3] - B[3])
        if dist < best:
            best, nb = dist, c
    if best > 0.5:
        raise EditError('the line moved')
    vb = _vertex_at(work, name, nb, snap)
    if vb < 0:
        raise EditError('cannot put a point where the line meets the outline')
    if va == vb:
        raise EditError('the line only touches the outline')
    before = set(work['panels'])
    cut(work, name, va, vb)
    made = sorted(n for n in work['panels'] if n not in before)
    for n in made:
        if panel_area(work['panels'][n]) < min_area:
            raise EditError('a piece would be too small')
    pat.clear()
    pat.update(work)
    return made


def cut_line(pattern, panel, x0, y0, x1, y1, min_area=4.0):
    """Cuts a panel in two along the line through (x0, y0) and (x1, y1) (panel coordinates, cm). Returns the names of the two new panels;
    raises EditError (pattern unchanged) when it cannot."""
    if panel not in pattern.get('panels', {}):
        raise EditError('no such panel: ' + panel)
    p0, p1 = np.array([x0, y0], dtype=float), np.array([x1, y1], dtype=float)
    if np.linalg.norm(p1 - p0) < 0.5:
        raise EditError('draw a longer line')
    cs = _line_crossings(pattern['panels'][panel], p0, p1)
    if len(cs) < 2:
        raise EditError('the line does not cross the piece')
    mid = np.linalg.norm(p1 - p0) / 2
    order = [i for i in range(0, len(cs) - 1, 2) if cs[i][2] <= mid <= cs[i + 1][2]]
    order += [i for i in range(0, len(cs) - 1, 2) if i not in order]
    last = EditError('the line does not cross the piece')
    try:
        for i in order:
            if cs[i + 1][2] - cs[i][2] < 1.0:
                continue
            try:
                return _cut_between(pattern, panel, p0, p1, cs[i], cs[i + 1], min_area)
            except EditError as ex:  # a refusal: the next stretch of the line
                last = ex
    except EditError:
        raise
    except Exception as ex:  # anything else ends the cut, as the C++ catch does
        raise EditError(str(ex))
    raise last


class _Mix64:
    """splitmix64: the same numbers as the C++ engine for the same seed"""
    def __init__(self, seed):
        self.s = seed & MASK64

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def uniform(self):
        return (self.next() >> 11) * (1.0 / 9007199254740992.0)


def apply_patchwork(pattern, design):
    """Applies the design's patchwork group to the pattern dict in place. Returns the number of cuts made."""
    if not isinstance(design, dict) or 'patchwork' not in design or 'panels' not in pattern:
        return 0
    pw = design['patchwork']

    def val(k, default):
        if k in pw and isinstance(pw[k], dict) and pw[k].get('v') is not None:
            return pw[k]['v']
        return default

    style = str(val('style', 'none'))
    if style not in ('triangles', 'shards', 'strips'):
        return 0
    size = min(max(float(val('piece_size', 15.0)), 4.0), 80.0)
    angle = float(val('angle', 0.0)) * math.pi / 180.0
    seed = max(0, int(val('seed', 0)))
    parts = str(val('parts', 'all'))
    side = str(val('side', 'both'))
    mirror = bool(val('mirror', True))

    # the pieces to cut, and how each maps the front view (world x, y) into its own plane
    todo = []  # (name, inverse of M, t, centre x)
    lo, hi = np.array([1e9, 1e9]), np.array([-1e9, -1e9])
    trims = ('cuff', 'wb', 'collar', 'lapel', 'hood', 'band', 'pocket', 'strap', 'facing', 'placket')
    for n in sorted(pattern['panels']):
        if any(w in n for w in trims):
            continue
        if parts == 'torso' and 'torso' not in n:
            continue
        if parts == 'sleeves' and 'sleeve' not in n:
            continue
        if parts == 'skirt' and not ('skirt' in n or 'pant' in n or 'leg' in n):
            continue
        p = pattern['panels'][n]
        rot = _rotation(p)
        ex, ey = rot.apply([1.0, 0.0, 0.0]), rot.apply([0.0, 1.0, 0.0])
        m = np.array([[ex[0], ey[0]], [ex[1], ey[1]]])
        det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
        if abs(det) < 0.5:
            continue
        t = np.array([float(p['translation'][0]), float(p['translation'][1])])
        cx = 0.0
        for i in range(len(p['vertices'])):
            w = m.dot(_vtx(p, i)) + t
            cx += w[0]
            lo = np.minimum(lo, w)
            hi = np.maximum(hi, w)
        cx /= max(1, len(p['vertices']))
        if side == 'left' and cx < 2:
            continue
        if side == 'right' and cx > -2:
            continue
        inv = np.array([[m[1, 1], -m[0, 1]], [-m[1, 0], m[0, 0]]]) / det
        todo.append((n, inv, t, cx))
    if not todo:
        return 0

    # the lines in the front view; tag +1: only the +x side of the garment, -1: only the -x side, 0: both
    lines = []
    mid = (lo + hi) / 2
    reach = np.linalg.norm(hi - lo)

    def add(p, d):
        if mirror:
            lines.append((p, d, +1))
            lines.append((np.array([-p[0], p[1]]), np.array([-d[0], d[1]]), -1))
        else:
            lines.append((p, d, 0))

    def family(a, spacing):
        d = np.array([math.cos(a), math.sin(a)])
        nn = np.array([-d[1], d[0]])
        origin = np.array([0.0, mid[1]])
        k = int(math.ceil(reach / spacing)) + 1
        for i in range(-k, k + 1):
            add(origin + nn * (i * spacing), d)

    if style == 'strips':
        family(angle + math.pi / 2, size)
    elif style == 'triangles':
        h = size * math.sqrt(3.0) / 2
        for f in range(3):
            family(angle + f * math.pi / 3, h)
    else:
        rng = _Mix64(seed)
        area = (hi[0] - lo[0]) * (hi[1] - lo[1])
        x = area / (size * size) / (2.0 if mirror else 1.0)
        n = min(max(int(math.floor(x + 0.5)), 2), 40)  # std::lround
        for _ in range(n):
            px = rng.uniform() * hi[0] if mirror else lo[0] + rng.uniform() * (hi[0] - lo[0])
            py = lo[1] + rng.uniform() * (hi[1] - lo[1])
            a = rng.uniform() * math.pi
            add(np.array([px, py]), np.array([math.cos(a), math.sin(a)]))

    min_area = max(6.0, 0.12 * size * size)
    max_cuts = 160
    cuts = 0
    log = os.environ.get('GG_PATCHWORK_LOG') is not None  # every cut tried, to compare with the C++ engine
    for name, inv, t, cx in todo:
        current = [name]
        for li, (p, d, tag) in enumerate(lines):
            if tag > 0 and cx < -2:
                continue
            if tag < 0 and cx > 2:
                continue
            a = inv.dot(p - d * reach * 2 - t)
            b = inv.dot(p + d * reach * 2 - t)
            nxt = []
            stack = list(reversed(current))
            while stack and cuts < max_cuts:
                piece = stack.pop()
                try:
                    made = cut_line(pattern, piece, a[0], a[1], b[0], b[1], min_area)
                    if log:
                        print(f'pw {piece} line {li}: cut', file=sys.stderr)
                    cuts += 1
                    for m_ in reversed(made):
                        stack.append(m_)
                except EditError as ex:
                    if log:
                        print(f'pw {piece} line {li}: {ex}', file=sys.stderr)
                    nxt.append(piece)
            nxt.extend(stack)
            current = nxt
    return cuts
