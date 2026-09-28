import os
import sys
import math
import time
from collections import deque
from heapq import heappush, heappop

POD_COST = 1000
TP_COST = 5000
POD_CAP = 10
DAYS = 20
MAX_DEG = 5
MAX_POD_ID = 500
INF = 1 << 30

# Tunable parameters (overridable through environment variables for local benchmarking).
HOP_COST = int(os.environ.get('HOP', '400'))  # Dijkstra penalty per hop
clock = time.process_time if os.environ.get('CPU_CLOCK') else time.perf_counter
VERBOSE = bool(os.environ.get('VERBOSE'))
TIME_SCALE = float(os.environ.get("TIME_SCALE", "1"))
FIRST_TURN_BUDGET = 0.65 * TIME_SCALE
TURN_BUDGET = 0.33 * TIME_SCALE
REGEN_EVERY = int(os.environ.get('REGEN', '6'))
NEW_EDGE_POD = float(os.environ.get('NEP', '300'))
EXT_MAX_RES = float(os.environ.get('EXTR', '40000'))
IMPROVE_K = int(os.environ.get('IMPK', '2'))
IMPROVE_EST = float(os.environ.get('IMPE', '60'))
THETA0 = float(os.environ.get('THETA', '8'))
THETA_WIN = int(os.environ.get('THWIN', '3'))
THETA_IDLE = float(os.environ.get('THIDLE', '0.25'))
HUB = int(os.environ.get('HUB', '1'))
HUB_K = int(os.environ.get('HUBK', '3'))
SAVE_F = float(os.environ.get('SAVEF', '1'))
SAVE_H = int(os.environ.get('SAVEH', '3'))
DESTROY_IDLE = int(os.environ.get('DESTROY', '1'))
BAL_W = float(os.environ.get('BALW', '0'))
BAL_A0 = int(os.environ.get('BALA0', '20'))
TP_WAIT_K = int(os.environ.get('TPW', '0'))
STUCK_K = int(os.environ.get('STK', '0'))
STUCK_MAXPOS = int(os.environ.get('STMP', '30'))
COV = float(os.environ.get('COV', '0'))
ALPHA = float(os.environ.get('ALPHA', '0.3'))


def debug_print(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True)


def tkey(a, b):
    return (a, b) if a < b else (b, a)


class Net:
    """Mutable transport network: tubes, teleporters and pods."""

    def __init__(self, n):
        self.adj = [[] for _ in range(n)]
        self.cap = {}
        self.tele = {}  # entrance -> exit
        self.tele_used = set()
        self.pods = {}  # id -> route
        self.pod_list = None
        self.succ = None

    def grow(self, n):
        while len(self.adj) < n:
            self.adj.append([])

    def add_tube(self, a, b, cap=1):
        self.cap[tkey(a, b)] = cap
        self.adj[a].append(b)
        self.adj[b].append(a)

    def remove_tube(self, a, b):
        del self.cap[tkey(a, b)]
        self.adj[a].remove(b)
        self.adj[b].remove(a)

    def add_tele(self, a, b):
        self.tele[a] = b
        self.tele_used.add(a)
        self.tele_used.add(b)

    def remove_tele(self, a, b):
        del self.tele[a]
        self.tele_used.discard(a)
        self.tele_used.discard(b)

    def add_pod(self, pid, route):
        self.pods[pid] = route
        self.pod_list = None
        self.succ = None

    def remove_pod(self, pid):
        del self.pods[pid]
        self.pod_list = None
        self.succ = None

    def get_succ(self):
        if self.succ is None:
            succ = {}
            for moves, L1, loop in self.get_pod_list():
                for a, b, k in moves:
                    lst = succ.get(a)
                    if lst is None:
                        succ[a] = [b]
                    elif b not in lst:
                        lst.append(b)
            self.succ = succ
        return self.succ

    def get_pod_list(self):
        if self.pod_list is None:
            lst = []
            for pid in sorted(self.pods):
                route = self.pods[pid]
                if len(route) > DAYS + 1:
                    route = route[:DAYS + 1]
                loop = len(route) > 1 and route[0] == route[-1]
                moves = [(route[j], route[j + 1], tkey(route[j], route[j + 1])) for j in range(len(route) - 1)]
                lst.append((moves, len(moves), loop))
            self.pod_list = lst
        return self.pod_list


class Game:
    def __init__(self):
        self.X = []
        self.Y = []
        self.T = []
        self.pads = {}
        self.modules_by_type = {}
        self.types_needed = []
        self.groups_tpl = []
        self.month = 0
        self.pair_ok = {}  # candidate pair -> validity (permanent once False)
        self.pair_len = {}
        self.tube_list = []
        self.net = Net(0)
        self.resources = 0
        self.prev_resources_end = None
        self.income_est = 0
        self.pad_months = []

    # ------------------------------------------------------------------ input
    def read_turn(self):
        self.resources = int(input())
        self.t0 = clock()
        self.month += 1
        n_routes = int(input())
        routes = []
        for _ in range(n_routes):
            a, b, c = map(int, input().split())
            routes.append((a, b, c))
        n_pods = int(input())
        pods = {}
        for _ in range(n_pods):
            parts = list(map(int, input().split()))
            pods[parts[0]] = parts[2:]
        n_new = int(input())
        new_ids = []
        for _ in range(n_new):
            parts = list(map(int, input().split()))
            typ, bid, x, y = parts[0], parts[1], parts[2], parts[3]
            while len(self.X) <= bid:
                self.X.append(0)
                self.Y.append(0)
                self.T.append(-1)
            self.X[bid] = x
            self.Y[bid] = y
            self.T[bid] = typ
            if typ == 0:
                self.pads[bid] = parts[5:]
            else:
                self.modules_by_type.setdefault(typ, []).append(bid)
            new_ids.append(bid)

        n = len(self.X)
        net = Net(n)
        for a, b, c in routes:
            if c == 0:
                net.add_tele(a, b)
            else:
                net.add_tube(a, b, c)
        for pid, route in pods.items():
            net.add_pod(pid, route)
        self.net = net
        self.deg = [len(net.adj[i]) for i in range(n)]

        # astronaut groups (sorted by boarding priority)
        tpl = []
        types = set()
        for p, lst in self.pads.items():
            first = {}
            cnt = {}
            for i, t in enumerate(lst):
                if t not in first:
                    first[t] = i
                cnt[t] = cnt.get(t, 0) + 1
            for t in first:
                tpl.append((p * 1000 + first[t], p, t, cnt[t]))
                types.add(t)
        tpl.sort()
        self.groups_tpl = tpl
        self.types_needed = sorted(types)

        # new tubes (from last turn) -> invalidate crossing pairs
        known = set(self.tube_list)
        for a, b, c in routes:
            if c != 0 and tkey(a, b) not in known:
                self.register_tube(a, b)
        if new_ids:
            self.update_pairs(new_ids)
        self.pad_months.append(any(self.T[b] == 0 for b in new_ids))

        if self.prev_resources_end is not None:
            self.income_est = max(0, self.resources - (self.prev_resources_end * 11) // 10)

    # --------------------------------------------------------------- geometry
    def on_segment(self, p, a, b):
        X, Y = self.X, self.Y
        px, py = X[p], Y[p]
        ax, ay, bx, by = X[a], Y[a], X[b], Y[b]
        if px < min(ax, bx) or px > max(ax, bx) or py < min(ay, by) or py > max(ay, by):
            return False
        return (bx - ax) * (py - ay) - (by - ay) * (px - ax) == 0

    def cross(self, a, b, c, d):
        X, Y = self.X, self.Y
        ax, ay, bx, by = X[a], Y[a], X[b], Y[b]
        cx, cy, dx, dy = X[c], Y[c], X[d], Y[d]
        o1 = (cy - ay) * (bx - ax) - (by - ay) * (cx - ax)
        o2 = (dy - ay) * (bx - ax) - (by - ay) * (dx - ax)
        if (o1 > 0 and o2 > 0) or (o1 < 0 and o2 < 0) or o1 == 0 or o2 == 0:
            return False
        o3 = (ay - cy) * (dx - cx) - (dy - cy) * (ax - cx)
        o4 = (by - cy) * (dx - cx) - (dy - cy) * (bx - cx)
        return (o3 > 0 > o4) or (o3 < 0 < o4)

    def tube_cost(self, a, b):
        return int(math.hypot(self.X[a] - self.X[b], self.Y[a] - self.Y[b]) * 10)

    def register_tube(self, a, b):
        k = tkey(a, b)
        self.tube_list.append(k)
        X, Y = self.X, self.Y
        minx, maxx = min(X[a], X[b]), max(X[a], X[b])
        miny, maxy = min(Y[a], Y[b]), max(Y[a], Y[b])
        pair_ok = self.pair_ok
        for p, ok in pair_ok.items():
            if not ok:
                continue
            c, d = p
            if max(X[c], X[d]) < minx or min(X[c], X[d]) > maxx or max(Y[c], Y[d]) < miny or min(Y[c], Y[d]) > maxy:
                continue
            if self.cross(a, b, c, d):
                pair_ok[p] = False

    def pair_valid_now(self, a, b):
        """Full check (buildings + tubes) for a new pair."""
        X, Y = self.X, self.Y
        minx, maxx = min(X[a], X[b]), max(X[a], X[b])
        miny, maxy = min(Y[a], Y[b]), max(Y[a], Y[b])
        for p in range(len(X)):
            if p == a or p == b or self.T[p] < 0:
                continue
            px, py = X[p], Y[p]
            if px < minx or px > maxx or py < miny or py > maxy:
                continue
            if (X[b] - X[a]) * (py - Y[a]) - (Y[b] - Y[a]) * (px - X[a]) == 0:
                return False
        for c, d in self.tube_list:
            if c == a or c == b or d == a or d == b:
                continue
            if max(X[c], X[d]) < minx or min(X[c], X[d]) > maxx or max(Y[c], Y[d]) < miny or min(Y[c], Y[d]) > maxy:
                continue
            if self.cross(a, b, c, d):
                return False
        return True

    def update_pairs(self, new_ids):
        X, Y, T = self.X, self.Y, self.T
        n = len(X)
        # new buildings may block existing pairs
        for p in new_ids:
            for pr, ok in self.pair_ok.items():
                if ok and self.on_segment(p, pr[0], pr[1]):
                    self.pair_ok[pr] = False
        # candidate pairs: k nearest + nearest in each of 8 cones + pad -> nearest module of needed types
        K = 8
        ids = [i for i in range(n) if T[i] >= 0]
        wanted = set()
        for i in ids:
            xi, yi = X[i], Y[i]
            dl = []
            for j in ids:
                if j != i:
                    dx, dy = X[j] - xi, Y[j] - yi
                    dl.append((dx * dx + dy * dy, j, dx, dy))
            dl.sort()
            for _, j, _, _ in dl[:K]:
                wanted.add(tkey(i, j))
            cones = [None] * 8
            for d2, j, dx, dy in dl:
                c = int((math.atan2(dy, dx) + math.pi) / (2 * math.pi) * 8) % 8
                if cones[c] is None:
                    cones[c] = j
            for j in cones:
                if j is not None:
                    wanted.add(tkey(i, j))
            if T[i] == 0:
                for t in set(self.pads[i]):
                    best = None
                    for m in self.modules_by_type.get(t, []):
                        d2 = (X[m] - xi) ** 2 + (Y[m] - yi) ** 2
                        if best is None or d2 < best[0]:
                            best = (d2, m)
                    if best:
                        wanted.add(tkey(i, best[1]))
        for pr in wanted:
            if pr not in self.pair_ok:
                self.pair_ok[pr] = self.pair_valid_now(pr[0], pr[1])
                self.pair_len[pr] = self.tube_cost(pr[0], pr[1])

    # ------------------------------------------------------------- simulation
    def compute_dists(self, net):
        n = len(self.X)
        adj = net.adj
        rtele = {b: a for a, b in net.tele.items()}
        dists = {}
        for t in self.types_needed:
            d = [INF] * n
            dq = deque()
            for m in self.modules_by_type.get(t, []):
                d[m] = 0
                dq.append(m)
            while dq:
                v = dq.popleft()
                dv = d[v]
                u = rtele.get(v)
                if u is not None and d[u] > dv:
                    d[u] = dv
                    dq.appendleft(u)
                dv1 = dv + 1
                for u in adj[v]:
                    if d[u] > dv1:
                        d[u] = dv1
                        dq.append(u)
            dists[t] = d
        return dists

    def update_dists(self, net, base_dists, ops):
        """Distances after adding the tubes/teleporters of ops (net already modified)."""
        adj = net.adj
        rtele = None
        seeds = []
        for op in ops:
            if op[0] == 'T':
                seeds.append((op[1], op[2], 1))
                seeds.append((op[2], op[1], 1))
            elif op[0] == 'X':
                seeds.append((op[2], op[1], 0))
        if any(w == 0 for _, _, w in seeds) or net.tele:
            rtele = {b: a for a, b in net.tele.items()}
        out = {}
        for t, d0 in base_dists.items():
            d = None
            dq = None
            for src, dst, w in seeds:
                nd = d0[src] + w if d is None else d[src] + w
                cur = d0[dst] if d is None else d[dst]
                if nd < cur:
                    if d is None:
                        d = d0[:]
                        dq = deque()
                    d[dst] = nd
                    dq.append(dst)
            if d is None:
                out[t] = d0
                continue
            while dq:
                v = dq.popleft()
                dv = d[v]
                if rtele is not None:
                    u = rtele.get(v)
                    if u is not None and d[u] > dv:
                        d[u] = dv
                        dq.append(u)
                dv1 = dv + 1
                for u in adj[v]:
                    if d[u] > dv1:
                        d[u] = dv1
                        dq.append(u)
            out[t] = d
        return out

    def simulate(self, net, detail=False, dists=None):
        T = self.T
        if dists is None:
            dists = self.compute_dists(net)
        tele = net.tele
        at = {}  # building -> list of [key, type, count] sorted by key
        dead = []
        succ = net.get_succ()

        def movable(p, d):
            dp = d[p]
            ex = tele.get(p)
            if ex is not None and d[ex] < INF and d[ex] <= dp:
                return True
            if dp >= INF:
                return False
            for b in succ.get(p, ()):
                if d[b] < dp:
                    return True
            return False

        for key, p, t, c in self.groups_tpl:
            if not movable(p, dists[t]):
                if detail:
                    dead.append((p, t, c))
                continue
            lst = at.get(p)
            if lst is None:
                at[p] = [[key, t, c]]
            else:
                lst.append([key, t, c])
        alloc = {}
        score = 0
        pods = net.get_pod_list()
        npods = len(pods)
        idx = [0] * npods
        cap = net.cap
        full = {} if detail else None
        riders = [0] * npods
        wait = {}
        for day in range(DAYS):
            if not at:
                break
            if tele:
                moves = []
                for e, ex in tele.items():
                    glist = at.get(e)
                    if glist is None:
                        continue
                    keep = []
                    for g in glist:
                        dt = dists[g[1]]
                        de = dt[ex]
                        if de < INF and de <= dt[e]:
                            moves.append((ex, g))
                        else:
                            keep.append(g)
                    if keep:
                        at[e] = keep
                    else:
                        del at[e]
                dirty = set()
                for ex, g in moves:
                    t = g[1]
                    if T[ex] == t:
                        n = g[2]
                        a = alloc.get(ex, 0)
                        alloc[ex] = a + n
                        score += (50 - day) * n
                        if a < 50:
                            up = a + n if a + n < 50 else 50
                            score += (101 - a - up) * (up - a) // 2
                    elif not movable(ex, dists[t]):
                        if detail:
                            dead.append((ex, t, g[2]))
                    else:
                        lst = at.get(ex)
                        if lst is None:
                            at[ex] = [g]
                        else:
                            lst.append(g)
                            dirty.add(ex)
                for b in dirty:
                    self._merge(at, b)
            used = {}
            leaving = {}
            for i in range(npods):
                moves, L1, loop = pods[i]
                ci = idx[i]
                if ci < L1:
                    a, b, k = moves[ci]
                    u = used.get(k, 0)
                    if u < cap[k]:
                        used[k] = u + 1
                        slot = [b, POD_CAP, k, i]
                        lst = leaving.get(a)
                        if lst is None:
                            leaving[a] = [slot]
                        else:
                            lst.append(slot)
                        ci += 1
                        if loop and ci == L1:
                            ci = 0
                        idx[i] = ci
            if not leaving:
                if not tele:
                    break
                continue
            arrivals = []
            arr_day = 50 - (day + 1)
            for a, slots in leaving.items():
                glist = at.get(a)
                if glist is None:
                    continue
                keep = []
                for g in glist:
                    t = g[1]
                    dt = dists[t]
                    cur = dt[a]
                    rem = g[2]
                    for slot in slots:
                        sc = slot[1]
                        if sc > 0 and dt[slot[0]] < cur:
                            n = rem if rem < sc else sc
                            slot[1] = sc - n
                            rem -= n
                            dest = slot[0]
                            if T[dest] == t:
                                al = alloc.get(dest, 0)
                                alloc[dest] = al + n
                                score += arr_day * n
                                if al < 50:
                                    up = al + n if al + n < 50 else 50
                                    score += (101 - al - up) * (up - al) // 2
                            else:
                                arrivals.append((dest, g[0], t, n))
                            if full is not None:
                                riders[slot[3]] += n
                                if slot[1] == 0:
                                    full[slot[2]] = full.get(slot[2], 0) + 1
                            if rem == 0:
                                break
                    if rem:
                        g[2] = rem
                        keep.append(g)
                if keep:
                    at[a] = keep
                else:
                    del at[a]
            if arrivals:
                dirty = set()
                for dest, key, t, n in arrivals:
                    if not movable(dest, dists[t]):
                        if detail:
                            dead.append((dest, t, n))
                        continue
                    lst = at.get(dest)
                    if lst is None:
                        at[dest] = [[key, t, n]]
                    else:
                        lst.append([key, t, n])
                        dirty.add(dest)
                for b in dirty:
                    self._merge(at, b)
            if detail:
                for b, glist in at.items():
                    for g in glist:
                        k2 = (b, g[1])
                        wait[k2] = wait.get(k2, 0) + g[2]
        if detail:
            self.last_wait = wait
            groups = list(dead)
            for p, glist in at.items():
                for g in glist:
                    groups.append((p, g[1], g[2]))
            self.last_riders = dict(zip(sorted(net.pods), riders))
            return score, groups, alloc, full, dists
        return score

    @staticmethod
    def _merge(at, b):
        lst = at[b]
        lst.sort()
        out = [lst[0]]
        for g in lst[1:]:
            if g[0] == out[-1][0]:
                out[-1][2] += g[2]
            else:
                out.append(g)
        at[b] = out

    # -------------------------------------------------------------- candidates
    def free_pod_id(self, net, taken):
        for i in range(1, MAX_POD_ID + 1):
            if i not in net.pods and i not in taken:
                return i
        return None

    def cand_cost(self, net, ops):
        cost = 0
        for op in ops:
            if op[0] == 'T':
                cost += self.tube_cost(op[1], op[2])
            elif op[0] == 'P':
                cost += POD_COST
            elif op[0] == 'U':
                k = tkey(op[1], op[2])
                cost += (net.cap[k] + 1) * self.tube_cost(op[1], op[2])
            elif op[0] == 'X':
                cost += TP_COST
            elif op[0] == 'R':
                cost += POD_COST - 750
        return cost

    def apply(self, net, ops):
        """Apply ops to net; returns undo list or None if invalid (net restored)."""
        undo = []
        ok = True
        new_tubes = []
        for op in ops:
            kind = op[0]
            if kind == 'T':
                a, b = op[1], op[2]
                k = tkey(a, b)
                if k in net.cap or len(net.adj[a]) >= MAX_DEG or len(net.adj[b]) >= MAX_DEG or not self.pair_ok.get(k, False):
                    ok = False
                    break
                for c, d in new_tubes:
                    if self.cross(a, b, c, d):
                        ok = False
                        break
                if not ok:
                    break
                net.add_tube(a, b)
                new_tubes.append((a, b))
                undo.append(('T', a, b))
            elif kind == 'P':
                route = op[1]
                for i in range(len(route) - 1):
                    if tkey(route[i], route[i + 1]) not in net.cap:
                        ok = False
                        break
                if not ok:
                    break
                pid = self.free_pod_id(net, ())
                if pid is None:
                    ok = False
                    break
                net.add_pod(pid, route)
                undo.append(('P', pid))
            elif kind == 'U':
                k = tkey(op[1], op[2])
                if k not in net.cap:
                    ok = False
                    break
                net.cap[k] += 1
                undo.append(('U', k))
            elif kind == 'R':
                pid, route = op[1], op[2]
                if pid not in net.pods:
                    ok = False
                    break
                for i in range(len(route) - 1):
                    if tkey(route[i], route[i + 1]) not in net.cap:
                        ok = False
                        break
                if not ok:
                    break
                undo.append(('R', pid, net.pods[pid]))
                net.pods[pid] = route
                net.pod_list = None
                net.succ = None
            elif kind == 'X':
                a, b = op[1], op[2]
                if a in net.tele_used or b in net.tele_used or a == b:
                    ok = False
                    break
                net.add_tele(a, b)
                undo.append(('X', a, b))
        if not ok:
            self.undo(net, undo)
            return None
        return undo

    def undo(self, net, undo):
        for op in reversed(undo):
            kind = op[0]
            if kind == 'T':
                net.remove_tube(op[1], op[2])
            elif kind == 'P':
                net.remove_pod(op[1])
            elif kind == 'U':
                net.cap[op[1]] -= 1
            elif kind == 'R':
                net.pods[op[1]] = op[2]
                net.pod_list = None
                net.succ = None
            elif kind == 'X':
                net.remove_tele(op[1], op[2])

    def path_dijkstra(self, net, t, alloc=None):
        """Multi-source Dijkstra from modules of type t over existing + candidate edges.
        Returns (dist, pred) where pred[v] = next node towards target."""
        n = len(self.X)
        dist = [INF] * n
        nxt = [-1] * n
        heap = []
        for m in self.modules_by_type.get(t, []):
            d0 = 0
            if alloc is not None:
                d0 = BAL_W * max(0, alloc.get(m, 0) - BAL_A0)
            dist[m] = d0
            heappush(heap, (d0, m))
        adj = net.adj
        cand = self.cand_adj
        rtele = self.rtele
        while heap:
            d, v = heappop(heap)
            if d > dist[v]:
                continue
            u = rtele.get(v)
            if u is not None and dist[u] > d:
                dist[u] = d
                nxt[u] = v
                heappush(heap, (d, u))
            nd = d + HOP_COST
            for u in adj[v]:
                if nd < dist[u]:
                    dist[u] = nd
                    nxt[u] = v
                    heappush(heap, (nd, u))
            for u, c in cand[v]:
                nd2 = d + c
                if nd2 < dist[u]:
                    dist[u] = nd2
                    nxt[u] = v
                    heappush(heap, (nd2, u))
        return dist, nxt

    def source_dijkstra(self, net, s):
        """Single-source Dijkstra from s over existing + candidate edges. Returns (dist, prev)."""
        n = len(self.X)
        dist = [INF] * n
        prev = [-1] * n
        dist[s] = 0
        heap = [(0, s)]
        adj = net.adj
        cand = self.cand_adj
        tele = net.tele
        while heap:
            d, v = heappop(heap)
            if d > dist[v]:
                continue
            u = tele.get(v)
            if u is not None and dist[u] > d:
                dist[u] = d
                prev[u] = v
                heappush(heap, (d, u))
            nd = d + HOP_COST
            for u in adj[v]:
                if nd < dist[u]:
                    dist[u] = nd
                    prev[u] = v
                    heappush(heap, (nd, u))
            for u, c in cand[v]:
                nd2 = d + c
                if nd2 < dist[u]:
                    dist[u] = nd2
                    prev[u] = v
                    heappush(heap, (nd2, u))
        return dist, prev

    def build_cand_adj(self, net):
        n = len(self.X)
        cand = [[] for _ in range(n)]
        deg = [len(a) for a in net.adj]
        for pr, ok in self.pair_ok.items():
            if not ok:
                continue
            a, b = pr
            if deg[a] >= MAX_DEG or deg[b] >= MAX_DEG or pr in net.cap:
                continue
            c = self.pair_len[pr] + NEW_EDGE_POD + HOP_COST
            cand[a].append((b, c))
            cand[b].append((a, c))
        self.cand_adj = cand
        self.rtele = {b: a for a, b in net.tele.items()}

    def path_ops(self, net, path):
        """Ops for building a path (list of nodes from source) with shuttle pods on new edges."""
        ops = []
        new_edges = []
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            if tkey(a, b) not in net.cap and net.tele.get(a) != b:
                new_edges.append((i, a, b))
        if not new_edges:
            return []
        variants = []
        tubes = [('T', a, b) for _, a, b in new_edges]
        ops = list(tubes)
        for i, a, b in new_edges:
            ops.append(('P', (a, b, a) if i % 2 == 0 else (b, a, b)))
        variants.append(ops)
        if len(new_edges) >= 2 and len(path) - 1 <= 10:
            # one pod travelling the whole path back and forth (only through tubes)
            if all(tkey(path[i], path[i + 1]) in net.cap or any(e[0] == i for e in new_edges) for i in range(len(path) - 1)):
                route = tuple(path) + tuple(reversed(path[:-1]))
                variants.append(tubes + [('P', route)])
        # extend existing pod lines with detours over the new chains
        chains = []
        for i, a, b in new_edges:
            if chains and chains[-1][1] == i - 1:
                chains[-1][1] = i
            else:
                chains.append([i, i])
        ext = list(tubes)
        modified = {}
        extended = False
        for ci, cj in chains:
            chain = path[ci:cj + 2]
            best = None
            for attach, ch in ((chain[0], chain), (chain[-1], chain[::-1])):
                for pid in self.pods_at.get(attach, ()):
                    route = modified.get(pid, net.pods[pid])
                    if len(route) + 2 * (len(ch) - 1) <= DAYS + 1:
                        key = (len(route), pid)
                        if best is None or key < best[0]:
                            best = (key, pid, attach, ch, route)
            if best is None:
                for k in range(ci, cj + 1):
                    a, b = path[k], path[k + 1]
                    ext.append(('P', (a, b, a) if k % 2 == 0 else (b, a, b)))
                continue
            _, pid, attach, ch, route = best
            i = route.index(attach)
            modified[pid] = tuple(route[:i + 1]) + tuple(ch[1:]) + tuple(ch[-2::-1]) + tuple(route[i + 1:])
            extended = True
        if extended and self.allow_ext:
            ext += [('R', pid, r) for pid, r in modified.items()]
            variants.append(ext)
        return variants

    def hub_candidates(self, net, stuck, direct, add):
        X, Y, T = self.X, self.Y, self.T
        groups = [(pos, t, c) for (pos, t), c in stuck.items() if t in direct and direct[t][pos] < INF]
        if not groups:
            return
        # entrance candidates: pads near many stuck astronauts
        ent = []
        for p in self.pads:
            if p in net.tele_used:
                continue
            sc = 0.0
            for pos, t, c in groups:
                sc += c / (1.0 + math.hypot(X[pos] - X[p], Y[pos] - Y[p]) / 10.0)
            ent.append((sc, p))
        ent.sort(reverse=True)
        # exit candidates: modules close to modules of the stuck types
        need = {}
        for pos, t, c in groups:
            need[t] = need.get(t, 0) + c
        ext = []
        for t0, mods in self.modules_by_type.items():
            for m in mods:
                if m in net.tele_used:
                    continue
                sc = 0.0
                for t, c in need.items():
                    best = min(math.hypot(X[m] - X[o], Y[m] - Y[o]) for o in self.modules_by_type[t])
                    sc += c / (1.0 + best / 10.0)
                ext.append((sc, m))
        ext.sort(reverse=True)
        budget = self.cur_resources - TP_COST
        dX_cache = {}
        for _, e in ent[:HUB_K]:
            dE, prevE = self.source_dijkstra(net, e)
            for _, x in ext[:HUB_K]:
                if x not in dX_cache:
                    dX_cache[x] = self.source_dijkstra(net, x)
                dX, prevX = dX_cache[x]
                items = []
                for pos, t, c in groups:
                    if dE[pos] >= INF:
                        continue
                    best = None
                    for m in self.modules_by_type[t]:
                        if dX[m] < INF and (best is None or dX[m] < dX[best]):
                            best = m
                    if best is None:
                        continue
                    via = dE[pos] + dX[best]
                    if via < direct[t][pos]:
                        items.append((c / (via + 100.0), pos, t, best))
                if not items:
                    continue
                items.sort(reverse=True)
                ops_t = []
                ops_p = []
                seen = set()
                spent = 0
                n_ast = 0
                for _, pos, t, m in items:
                    path1 = [pos]
                    v = pos
                    while prevE[v] != -1 and len(path1) < 30:
                        v = prevE[v]
                        path1.append(v)
                    path2 = [m]
                    v = m
                    while prevX[v] != -1 and len(path2) < 30:
                        v = prevX[v]
                        path2.append(v)
                    path2.reverse()
                    add_t = []
                    add_p = []
                    extra = 0
                    for path in (path1, path2):
                        for i in range(len(path) - 1):
                            a, b = path[i], path[i + 1]
                            k = tkey(a, b)
                            if k in net.cap or k in seen or net.tele.get(a) == b:
                                continue
                            add_t.append(('T', a, b))
                            add_p.append(('P', (a, b, a) if i % 2 == 0 else (b, a, b)))
                            extra += self.pair_len.get(k, 0) + POD_COST
                            seen.add(k)
                    if spent + extra > budget:
                        for op in add_t:
                            seen.discard(tkey(op[1], op[2]))
                        continue
                    spent += extra
                    ops_t += add_t
                    ops_p += add_p
                    n_ast += stuck[(pos, t)]
                add([('X', e, x)] + ops_t + ops_p, n_ast * 80)

    def gen_candidates(self, net, base_detail):
        score, groups, alloc, full, dists = base_detail
        cands = {}

        def add(ops, est):
            if ops is None:
                return
            key = tuple(ops)
            if key in cands:
                return
            cost = self.cand_cost(net, ops)
            if cost <= 0:
                return
            cands[key] = (ops, cost, est)

        self.build_cand_adj(net)
        pods_at = {}
        for pid, route in net.pods.items():
            for v in set(route):
                pods_at.setdefault(v, []).append(pid)
        self.pods_at = pods_at
        # stuck astronauts
        stuck = {}
        for pos, t, c in groups:
            stuck[(pos, t)] = stuck.get((pos, t), 0) + c
        by_type = {}
        direct = {}
        for (pos, t), c in stuck.items():
            by_type.setdefault(t, []).append((pos, c))
        for t, lst in by_type.items():
            dist, nxt = self.path_dijkstra(net, t, alloc if BAL_W > 0 else None)
            direct[t] = dist
            for pos, c in lst:
                if dist[pos] >= INF:
                    continue
                path = [pos]
                v = pos
                while nxt[v] != -1 and len(path) < 30:
                    v = nxt[v]
                    path.append(v)
                if self.T[path[-1]] != t:
                    continue
                for ops in self.path_ops(net, path):
                    add(ops, c * 100)

        # alternative targets for stuck astronauts (small instances only)
        positions = set(pos for pos, t in stuck)
        if STUCK_K > 0 and len(positions) <= STUCK_MAXPOS:
            for pos in positions:
                dist, prev = self.source_dijkstra(net, pos)
                for (p2, t), c in stuck.items():
                    if p2 != pos:
                        continue
                    mods = [m for m in self.modules_by_type.get(t, []) if dist[m] < INF]
                    mods.sort(key=lambda m: dist[m])
                    for m in mods[:STUCK_K]:
                        path = [m]
                        v = m
                        while prev[v] != -1 and len(path) < 30:
                            v = prev[v]
                            path.append(v)
                        path.reverse()
                        for ops in self.path_ops(net, path):
                            add(ops, c * 90)

        # improvements: shortcuts / second modules for each pad and type
        if IMPROVE_K > 0:
            for p, lst in self.pads.items():
                cnt = {}
                for t in lst:
                    cnt[t] = cnt.get(t, 0) + 1
                dist, prev = self.source_dijkstra(net, p)
                for t, c in cnt.items():
                    mods = [m for m in self.modules_by_type.get(t, []) if dist[m] < INF]
                    mods.sort(key=lambda m: dist[m])
                    taken = 0
                    for m in mods:
                        path = [m]
                        v = m
                        while prev[v] != -1 and len(path) < 30:
                            v = prev[v]
                            path.append(v)
                        path.reverse()
                        variants = self.path_ops(net, path)
                        if variants:
                            for ops in variants:
                                add(ops, c * IMPROVE_EST)
                            taken += 1
                            if taken >= IMPROVE_K:
                                break

        if HUB and self.cur_resources >= TP_COST and stuck:
            self.hub_candidates(net, stuck, direct, add)

        # teleporters
        for p, lst in self.pads.items():
            if p in net.tele_used:
                continue
            cnt = {}
            for t in lst:
                cnt[t] = cnt.get(t, 0) + 1
            for t, c in sorted(cnt.items(), key=lambda x: -x[1])[:2]:
                best = None
                for m in self.modules_by_type.get(t, []):
                    if m in net.tele_used:
                        continue
                    key = (alloc.get(m, 0), m)
                    if best is None or key < best:
                        best = key
                if best is not None:
                    add([('X', p, best[1])], c * 60 + (len(lst) - c) * 10)

        # teleporters from congested buildings
        if TP_WAIT_K > 0:
            wb = {}
            for (b, t), w in self.last_wait.items():
                if b in net.tele_used:
                    continue
                wb.setdefault(b, {})
                wb[b][t] = wb[b].get(t, 0) + w
            tops = sorted(wb.items(), key=lambda kv: -sum(kv[1].values()))[:TP_WAIT_K]
            for b, tw in tops:
                for t, w in sorted(tw.items(), key=lambda x: -x[1])[:2]:
                    best = None
                    for m in self.modules_by_type.get(t, []):
                        if m in net.tele_used or m == b:
                            continue
                        key = (alloc.get(m, 0), m)
                        if best is None or key < best:
                            best = key
                    if best is not None:
                        add([('X', b, best[1])], w * 5)

        # capacity on saturated tubes
        for k, f in full.items():
            if f < 2:
                continue
            a, b = k
            npods = 0
            for moves, L1, loop in net.get_pod_list():
                for mv in moves:
                    if mv[2] == k:
                        npods += 1
                        break
            for route in ((a, b, a), (b, a, b)):
                if npods >= net.cap[k]:
                    add([('U', a, b), ('P', route)], f * 100)
                else:
                    add([('P', route)], f * 100)
        return cands

    # ------------------------------------------------------------------- plan
    def plan(self):
        budget_time = FIRST_TURN_BUDGET if self.month == 1 else TURN_BUDGET
        deadline = self.t0 + budget_time
        net = self.net
        actions = []
        self.actions = actions
        resources = self.resources
        self.allow_ext = resources < EXT_MAX_RES
        self.cur_resources = resources
        m_rem = DAYS + 1 - self.month
        theta = THETA0 * (m_rem - 1) / (DAYS - 1)
        recent = self.pad_months[-THETA_WIN:]
        if self.month > 1 and not any(recent):
            theta *= THETA_IDLE
        base = self.simulate(net, detail=True)
        cur_score = base[0]
        if DESTROY_IDLE:
            idle = [pid for pid, r in self.last_riders.items() if r == 0]
            removed = False
            for pid in idle:
                if clock() > self.t0 + budget_time * 0.2:
                    break
                route = net.pods[pid]
                net.remove_pod(pid)
                if self.simulate(net) >= cur_score:
                    actions.append(f"DESTROY {pid}")
                    resources += 750
                    removed = True
                else:
                    net.add_pod(pid, route)
            if removed:
                base = self.simulate(net, detail=True)
                cur_score = base[0]
        if SAVE_F > 0 and m_rem > 2:
            theta = max(theta, self.saving_threshold(net, base, resources, m_rem, self.t0 + budget_time * 0.3) * SAVE_F)
        n_eval = 0
        cache = {}  # key -> (gain, version evaluated at)
        cand_nodes = {}
        pick_nodes = []  # nodes touched by pick i (version i -> i+1)
        version = 0
        picks_since_gen = 0
        heap = None
        cands = None
        dirty = False

        def independent(key, ver):
            nodes = cand_nodes[key]
            for i in range(ver, version):
                if not nodes.isdisjoint(pick_nodes[i]):
                    return False
            return True

        while clock() < deadline:
            if heap is None or picks_since_gen >= REGEN_EVERY:
                if dirty:
                    base = self.simulate(net, detail=True)
                    cur_score = base[0]
                    dirty = False
                cands = self.gen_candidates(net, base)
                heap = []
                for key, (ops, cost, est) in cands.items():
                    if key not in cand_nodes:
                        cand_nodes[key] = self.ops_nodes(net, ops)
                    prio = (cache[key][0] if key in cache else est) / (cost + ALPHA * resources)
                    heappush(heap, (-prio, cost, 1, key))
                picks_since_gen = 0
                fresh_gen = True
            else:
                fresh_gen = False
            chosen = None
            best_fresh = None
            while heap:
                if clock() >= deadline:
                    chosen = best_fresh
                    break
                negp, cost, _, key = heappop(heap)
                ops = cands[key][0]
                cost = self.cand_cost(net, ops)
                if cost > resources:
                    continue
                c = cache.get(key)
                if c is not None and independent(key, c[1]):
                    if c[0] <= 0 or c[0] * m_rem < theta * cost:
                        continue
                    ratio = c[0] / (cost + ALPHA * resources)
                    if -negp <= ratio * 1.0000001:
                        chosen = key
                        break
                    heappush(heap, (-ratio, cost, 0, key))
                    continue
                if dirty:
                    base = self.simulate(net, detail=True)
                    cur_score = base[0]
                    dirty = False
                newly = 0
                if COV > 0:
                    for op in ops:
                        if op[0] == 'T':
                            newly += (not net.adj[op[1]]) + (not net.adj[op[2]])
                undo = self.apply(net, ops)
                if undo is None:
                    continue
                same_graph = True
                for op in ops:
                    if op[0] == 'T' or op[0] == 'X':
                        same_graph = False
                        break
                sc = self.simulate(net, dists=base[4] if same_graph else self.update_dists(net, base[4], ops))
                n_eval += 1
                self.undo(net, undo)
                gain = sc - cur_score
                if gain > 0 and newly:
                    gain += COV * newly
                cache[key] = (gain, version)
                if gain > 0 and gain * m_rem >= theta * cost:
                    ratio = gain / (cost + ALPHA * resources)
                    heappush(heap, (-ratio, cost, 0, key))
                    if best_fresh is None or ratio > best_fresh_ratio:
                        best_fresh = key
                        best_fresh_ratio = ratio
            if chosen is None:
                if fresh_gen or clock() >= deadline:
                    break
                heap = None
                continue
            ops = cands[chosen][0]
            cost = self.cand_cost(net, ops)
            nodes = self.ops_nodes(net, ops)
            undo = self.apply(net, ops)
            if undo is None:
                continue
            resources -= cost
            actions.extend(self.ops_to_actions(net, undo))
            for op in undo:
                if op[0] == 'T':
                    self.register_tube(op[1], op[2])
            if VERBOSE:
                debug_print(f"  pick {chosen} cost {cost} gain~{cache[chosen][0]}")
            dirty = True
            cache.pop(chosen, None)
            pick_nodes.append(nodes)
            version += 1
            picks_since_gen += 1
        if dirty:
            cur_score = self.simulate(net)
        debug_print(f"month {self.month} res {self.resources} predicted {cur_score} evals {n_eval} left {resources} time {clock() - self.t0:.3f}")
        self.prev_resources_end = resources
        return actions

    def saving_threshold(self, net, base, resources, m_rem, deadline):
        """Value (gain * months / cost) of the best teleporter-based plan affordable within a few months."""
        budgets = []
        fut = resources
        for k in range(SAVE_H):
            fut = fut * 11 // 10 + self.income_est
            budgets.append(fut)
        if budgets[-1] < TP_COST:
            return 0.0
        saved = self.cur_resources
        self.cur_resources = budgets[-1]
        cands = self.gen_candidates(net, base)
        self.cur_resources = saved
        best = 0.0
        cur = base[0]
        lst = sorted(cands.values(), key=lambda c: -c[2] / c[1])
        for ops, cost, est in lst:
            if clock() > deadline:
                break
            if cost <= resources or ops[0][0] != 'X':
                continue
            months = None
            for k, b in enumerate(budgets):
                if b >= cost:
                    months = k + 1
                    break
            if months is None or m_rem - months <= 0:
                continue
            undo = self.apply(net, ops)
            if undo is None:
                continue
            gain = self.simulate(net) - cur
            self.undo(net, undo)
            v = gain * (m_rem - months) / cost
            if v > best:
                best = v
        return best

    def ops_nodes(self, net, ops):
        s = set()
        for op in ops:
            k = op[0]
            if k == 'T' or k == 'U' or k == 'X':
                s.add(op[1])
                s.add(op[2])
            elif k == 'P':
                s.update(op[1])
            elif k == 'R':
                s.update(op[2])
                s.update(net.pods[op[1]])
        return s

    def ops_to_actions(self, net, undo):
        out = []
        for op in undo:
            if op[0] == 'T':
                out.append(f"TUBE {op[1]} {op[2]}")
            elif op[0] == 'U':
                out.append(f"UPGRADE {op[1][0]} {op[1][1]}")
            elif op[0] == 'X':
                out.append(f"TELEPORT {op[1]} {op[2]}")
            elif op[0] == 'P':
                route = net.pods[op[1]]
                out.append(f"POD {op[1]} " + " ".join(map(str, route)))
            elif op[0] == 'R':
                route = net.pods[op[1]]
                out.append(f"DESTROY {op[1]}")
                out.append(f"POD {op[1]} " + " ".join(map(str, route)))
        return out


def main():
    game = Game()
    while True:
        game.read_turn()
        game.actions = []
        try:
            actions = game.plan()
        except Exception as e:
            debug_print("error", repr(e))
            actions = game.actions
        print(";".join(actions) if actions else "WAIT", flush=True)


if __name__ == "__main__":
    main()
