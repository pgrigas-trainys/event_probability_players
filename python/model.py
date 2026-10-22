"""Skaičiavimų modelis Python kalba (tas pats kaip app/model.js).

Koeficientai -> tikimybės be maržos, normaliojo skirstinio įvertis kiekvienai rinkai,
abiejų komandų taškų modelis su koreliacija, rezultatų tinklelis ir sąlygų tikimybė.
Naudoja Python standartinę biblioteką. Jei įdiegta NumPy, rezultatų tinklelis ir sąlygos
skaičiuojami vektoriškai (visas tinklelis vienu kartu, ~10 kartų greičiau); be NumPy tas pats
skaičiuojama paprastais ciklais, rezultatas nesikeičia.

Paleidimas iš repozitorijos šaknies:
    python3 python/model.py --hcp data/hcp.csv --tot data/tot.csv \
        --home-name Fenerbahce --away-name Partizan \
        --cond "home>75" --cond "away>75" --cond "diff<0"

CSV failuose naudojami pirmi trys skaitiniai stulpeliai: linija, koeficientas A, koeficientas B.
Handikapui linija yra namų komandos handikapas (pvz. -11.5), A = namai, B = svečiai.
Sumoms ir komandų taškams A = daugiau, B = mažiau.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from statistics import NormalDist

try:
    import numpy as np
except ImportError:  # be NumPy (pvz. Brython) naudojami ciklai
    np = None

N = NormalDist()
SIGMA0 = {"hcp": 12.0, "tot": 16.0, "home": 10.0, "away": 10.0}


# ---------- koeficientai -> tikimybės ----------
@dataclass
class Row:
    k: float        # slenkstis: X > k (handikapui X = namai − svečiai, k = −linija)
    pa: float       # P(A) be maržos
    pb: float       # P(B) be maržos
    margin: float   # bukmekerio marža


def row_calc(market: str, line: float, a: float, b: float) -> Row | None:
    if not (a > 1 and b > 1) or not math.isfinite(line):
        return None
    ia, ib = 1 / a, 1 / b
    book = ia + ib
    return Row(k=-line if market == "hcp" else line, pa=ia / book, pb=ib / book, margin=book - 1)


# ---------- normaliojo skirstinio įvertis ----------
@dataclass
class Fit:
    mu: float
    sigma: float
    n: int
    assumed: bool   # True, kai σ neįvertinta (viena linija) ir paimta numatyta


def fit(market: str, rows: list[tuple[float, float, float]]) -> Fit | None:
    """P(X > k) = 1 − Φ((k − μ)/σ): z = Φ⁻¹(1 − p) tiesiškai priklauso nuo k, nuolydis = 1/σ."""
    calc = [c for c in (row_calc(market, *r) for r in rows) if c]
    if not calc:
        return None
    pts = [(c.k, N.inv_cdf(1 - min(0.99, max(0.01, c.pa)))) for c in calc]
    if len(pts) >= 2:
        n = len(pts)
        mk = sum(k for k, _ in pts) / n
        mz = sum(z for _, z in pts) / n
        sxx = sum((k - mk) ** 2 for k, _ in pts)
        if sxx > 0:
            slope = sum((k - mk) * (z - mz) for k, z in pts) / sxx
            if slope > 0.01:
                sigma = 1 / slope
                return Fit(mu=mk - mz * sigma, sigma=sigma, n=n, assumed=False)
    sigma = SIGMA0[market]
    mu = sum(k - z * sigma for k, z in pts) / len(pts)
    return Fit(mu=mu, sigma=sigma, n=len(pts), assumed=True)


def p_over(f: Fit, k: float) -> float:
    return 1 - N.cdf((k - f.mu) / f.sigma)


# ---------- rezultato modelis ----------
@dataclass
class ScoreModel:
    mh: float   # namų taškų vidurkis
    ma: float   # svečių taškų vidurkis
    sh: float   # namų σ
    sa: float   # svečių σ
    rho: float  # koreliacija tarp komandų taškų
    src: str
    rho_tot: float | None = None  # ρ iš komandų taškų σ ir sumos σ (palyginimui)
    rho_hcp: float | None = None  # ρ iš komandų taškų σ ir handikapo σ (palyginimui)


# Namų (H) ir svečių (A) taškai – dvimatis normalusis skirstinys su μh, μa, σh, σa, ρ.
# Kiekviena rinka duoda vidurkį ir dispersiją, tiesiškai priklausančius nuo nežinomųjų:
#   handikapas  H − A:  μh − μa,   σh² + σa² − 2c        (c = ρ·σh·σa)
#   suma        H + A:  μh + μa,   σh² + σa² + 2c
#   namų taškai H:      μh,        σh²
#   svečių      A:      μa,        σa²
# Visos turimos rinkos sujungiamos svertiniu mažiausių kvadratų metodu (vidurkiai: svoris n/σ²,
# dispersijos: n/σ⁴). Labai silpnos prielaidos σh = σa ir ρ = 0 lemia tik tai, ko rinkos neapibrėžia.
SCORE_ROWS = {
    "hcp": ((1, -1), (1, 1, -2), "handikapas"),
    "tot": ((1, 1), (1, 1, 2), "suma"),
    "home": ((1, 0), (1, 0, 0), "namų taškai"),
    "away": ((0, 1), (0, 1, 0), "svečių taškai"),
}


def _wls(rows: list[tuple[tuple[float, ...], float, float]]) -> list[float] | None:
    """Svertinis mažiausių kvadratų sprendinys (Gauso eliminacija); None, jei neapibrėžtas."""
    n = len(rows[0][0])
    A = [[0.0] * (n + 1) for _ in range(n)]
    for x, y, w in rows:
        for i in range(n):
            for j in range(n):
                A[i][j] += w * x[i] * x[j]
            A[i][n] += w * x[i] * y
    scale = max(abs(A[i][i]) for i in range(n)) or 1.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        if abs(A[p][c]) < 1e-12 * scale:
            return None
        A[c], A[p] = A[p], A[c]
        for r in range(n):
            if r != c:
                f = A[r][c] / A[c][c]
                for j in range(c, n + 1):
                    A[r][j] -= f * A[c][j]
    return [A[i][n] / A[i][i] for i in range(n)]


def score_model(F: dict[str, Fit | None]) -> ScoreModel | None:
    keys = [k for k in SCORE_ROWS if F.get(k)]
    if not keys:
        return None
    mean = _wls([(SCORE_ROWS[k][0], F[k].mu, F[k].n / F[k].sigma ** 2) for k in keys])
    if mean is None:
        return None
    v_rows = [(SCORE_ROWS[k][1], F[k].sigma ** 2, (1e-3 if F[k].assumed else 1.0) * F[k].n / F[k].sigma ** 4)
              for k in keys]
    w_max = max(r[2] for r in v_rows)
    v_rows += [((1, -1, 0), 0.0, 1e-6 * w_max), ((0, 0, 1), 0.0, 1e-6 * w_max)]
    vh, va, c = _wls(v_rows)
    sh, sa = math.sqrt(max(vh, 1.0)), math.sqrt(max(va, 1.0))
    rho = max(-0.95, min(0.95, c / (sh * sa)))
    rho_tot = rho_hcp = None
    real = lambda k: F.get(k) is not None and not F[k].assumed  # noqa: E731
    if real("home") and real("away"):
        vH, vA = F["home"].sigma ** 2, F["away"].sigma ** 2
        d = 2 * math.sqrt(vH * vA)
        if real("tot"):
            rho_tot = (F["tot"].sigma ** 2 - vH - vA) / d
        if real("hcp"):
            rho_hcp = (vH + vA - F["hcp"].sigma ** 2) / d
    return ScoreModel(mh=mean[0], ma=mean[1], sh=sh, sa=sa, rho=rho, src=" + ".join(SCORE_ROWS[k][2] for k in keys),
                      rho_tot=rho_tot, rho_hcp=rho_hcp)


def _grid_arrays(M: ScoreModel):
    """NumPy versija: masyvai H, A, P visiems sveikiems rezultatams per ±5σ."""
    rho = max(-0.95, min(0.95, M.rho))
    h = np.arange(max(0, math.floor(M.mh - 5 * M.sh)), math.ceil(M.mh + 5 * M.sh) + 1)
    a = np.arange(max(0, math.floor(M.ma - 5 * M.sa)), math.ceil(M.ma + 5 * M.sa) + 1)
    H, A = np.meshgrid(h, a, indexing="ij")
    x, y = (H - M.mh) / M.sh, (A - M.ma) / M.sa
    d = np.exp(-(x * x - 2 * rho * x * y + y * y) / (2 * (1 - rho * rho)))
    return H.ravel(), A.ravel(), (d / d.sum()).ravel()


def grid_cells(M: ScoreModel) -> list[tuple[int, int, float]]:
    """Visi sveiki rezultatai h : a per ±5σ su dvimačio normaliojo tankiu, sunormuoti iki 1."""
    if np is not None:
        H, A, P = _grid_arrays(M)
        return list(zip(H.tolist(), A.tolist(), P.tolist()))
    rho = max(-0.95, min(0.95, M.rho))
    cells, tot = [], 0.0
    for h in range(max(0, math.floor(M.mh - 5 * M.sh)), math.ceil(M.mh + 5 * M.sh) + 1):
        for a in range(max(0, math.floor(M.ma - 5 * M.sa)), math.ceil(M.ma + 5 * M.sa) + 1):
            x, y = (h - M.mh) / M.sh, (a - M.ma) / M.sa
            d = math.exp(-(x * x - 2 * rho * x * y + y * y) / (2 * (1 - rho * rho)))
            cells.append((h, a, d))
            tot += d
    return [(h, a, d / tot) for h, a, d in cells]


# ---------- sąlygos ----------
VARS = {"home": lambda h, a: h, "away": lambda h, a: a, "total": lambda h, a: h + a, "diff": lambda h, a: h - a}
OPS = {">": lambda x, y: x > y, ">=": lambda x, y: x >= y, "<": lambda x, y: x < y,
       "<=": lambda x, y: x <= y, "=": lambda x, y: x == y}


def condition_probability(M: ScoreModel, conds: list[tuple[str, str, float]]) -> float:
    """conds = [("home", ">", 75), ...], visos sujungtos „ir“."""
    for v, op, _ in conds:
        if v not in VARS or op not in OPS:
            raise ValueError(f"Netinkama sąlyga: {v} {op}")
    if np is not None:
        H, A, P = _grid_arrays(M)
        mask = np.ones(P.shape, dtype=bool)
        for v, op, x in conds:
            mask &= OPS[op](VARS[v](H, A), x)
        return float(P[mask].sum())
    return sum(w for h, a, w in grid_cells(M)
               if all(OPS[op](VARS[v](h, a), x) for v, op, x in conds))


def parse_cond(text: str) -> tuple[str, str, float]:
    m = re.fullmatch(r"\s*(home|away|total|diff)\s*(>=|<=|>|<|=)\s*(-?\d+(?:[.,]\d+)?)\s*", text)
    if not m:
        raise argparse.ArgumentTypeError(f"Sąlyga turi būti pvz. home>75, diff<0: {text!r}")
    return m.group(1), m.group(2), float(m.group(3).replace(",", "."))


# ---------- Dirichlet žaidėjų dalių modelis ----------
# Komandos taškai T (iš rezultato modelio) dalijami žaidėjams: dalys (S₁…S_k, S_kiti) ~ Dirichlet(c·m₁, …, c·m_k, c·m_kiti),
# žaidėjo taškai Xᵢ = Sᵢ·T. Vienam žaidėjui Sᵢ ~ Beta(c·mᵢ, c·(1−mᵢ)), nepriklausoma nuo T, todėl
# P(Xᵢ > linija) = E_T[1 − I_{linija/T}(c·mᵢ, c·(1−mᵢ))]. mᵢ parenkamas, kad ši tikimybė atitiktų rinką be maržos;
# c (koncentracija) nustatoma ranka arba įvertinama iš žaidėjų su keliomis linijomis. Tas pats kaip app/model.js.
def _betacf(a: float, b: float, x: float) -> float:
    tiny, qab, qap, qam = 1e-300, a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 301):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d; d = d if abs(d) > tiny else tiny
        c = 1 + aa / c; c = c if abs(c) > tiny else tiny
        d = 1 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d; d = d if abs(d) > tiny else tiny
        c = 1 + aa / c; c = c if abs(c) > tiny else tiny
        d = 1 / d; delta = d * c; h *= delta
        if abs(delta - 1) < 1e-13:
            break
    return h


def beta_cdf(x: float, a: float, b: float) -> float:
    """Reguliarizuota nepilnoji beta funkcija I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    return bt * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * _betacf(b, a, 1 - x) / b


def _beta_cdf_many(xs, a: float, b: float) -> list[float]:
    """beta_cdf daugeliui x. Taškų nedaug (41), todėl paprastas ciklas greitesnis už NumPy."""
    return [beta_cdf(x, a, b) for x in xs]


def _bisect(f, lo: float, hi: float, it: int = 60) -> float:
    for _ in range(it):
        mid = (lo + hi) / 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def beta_quantile(q: float, a: float, b: float) -> float:
    return _bisect(lambda x: beta_cdf(x, a, b) >= q, 0.0, 1.0)


def team_nodes(mu: float, sd: float, n: int = 41) -> tuple[list[float], list[float]]:
    """Komandos taškų normalusis skirstinys kaip svoriai taškuose ±4σ: (taškai, svoriai)."""
    ts, ws = [], []
    for i in range(n):
        z = -4 + 8 * i / (n - 1)
        t = mu + z * sd
        if t > 1:
            ts.append(t); ws.append(math.exp(-z * z / 2))
    tot = sum(ws)
    return ts, [w / tot for w in ws]


def share_p_over(m: float, c: float, L: float, nodes) -> float:
    ts, ws = nodes
    cdf = _beta_cdf_many([L / t for t in ts], c * m, c * (1 - m))
    return float(sum(w * (1 - f) for w, f in zip(ws, cdf)))


def _probit(p: float) -> float:
    return N.inv_cdf(min(0.995, max(0.005, p)))


def _golden(f, lo: float, hi: float, it: int = 60) -> float:
    r = (math.sqrt(5) - 1) / 2
    x1, x2 = hi - r * (hi - lo), lo + r * (hi - lo)
    f1, f2 = f(x1), f(x2)
    for _ in range(it):
        if f1 < f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - r * (hi - lo); f1 = f(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + r * (hi - lo); f2 = f(x2)
    return (lo + hi) / 2


def fit_share(lines: list[dict], c: float, nodes) -> tuple[float, float]:
    """lines = [{"L": linija, "p": rinkos P(daugiau)}] -> (vidutinė dalis m, paklaida probit skalėje)."""
    sse = lambda m: sum((_probit(share_p_over(m, c, l["L"], nodes)) - _probit(l["p"])) ** 2 for l in lines)  # noqa: E731
    m = math.exp(_golden(lambda u: sse(math.exp(u)), math.log(0.002), math.log(0.9), 45))
    return m, sse(m)


def _points_quantile(q: float, m: float, c: float, nodes, hi: float) -> float:
    ts, ws = nodes
    a, b = c * m, c * (1 - m)
    return _bisect(lambda x: sum(w * f for w, f in zip(ws, _beta_cdf_many([x / t for t in ts], a, b))) >= q, 0.0, hi, 40)


def dirichlet_team(team: dict, players: list[dict], c: float) -> dict:
    """team = {"mu", "sd"}; players = [{"id", "n", "lines": [{"L", "p"}]}] -> dalys, intervalai, modelio tikimybės."""
    nodes = team_nodes(team["mu"], team["sd"])
    hi = team["mu"] + 6 * team["sd"]
    res, sse = [], 0.0
    for pl in players:
        m, e = fit_share(pl["lines"], c, nodes)
        sse += e
        a, b = c * m, c * (1 - m)
        res.append({"id": pl["id"], "n": pl["n"], "m": m, "lo": beta_quantile(0.1, a, b), "hi": beta_quantile(0.9, a, b),
                    "mean": m * team["mu"], "ptsLo": _points_quantile(0.1, m, c, nodes, hi),
                    "ptsHi": _points_quantile(0.9, m, c, nodes, hi),
                    "lines": [{"L": l["L"], "p": l["p"], "pm": share_p_over(m, c, l["L"], nodes)} for l in pl["lines"]]})
    return {"c": c, "mu": team["mu"], "sd": team["sd"], "players": res, "rest": 1 - sum(r["m"] for r in res), "sse": sse}


def fit_concentration(groups: list[dict]) -> dict | None:
    """c, su kuriuo modelis geriausiai atkuria visas linijas (informatyvūs tik žaidėjai su ≥ 2 linijomis)."""
    multi = [(team_nodes(g["team"]["mu"], g["team"]["sd"]), p["lines"]) for g in groups for p in g["players"] if len(p["lines"]) >= 2]
    if not multi:
        return None
    obj = lambda c: sum(fit_share(lines, c, nodes)[1] for nodes, lines in multi)  # noqa: E731
    lo, step = math.log(3), (math.log(1500) - math.log(3)) / 15
    best = min((lo + step * i for i in range(16)), key=lambda u: obj(math.exp(u)))
    return {"c": math.exp(_golden(lambda u: obj(math.exp(u)), best - step, best + step, 20)), "players": len(multi)}


def player_condition_probability(M: ScoreModel, teams: dict, conds: list[tuple[str, str, float]],
                                 n: int = 40000, seed: int = 12345) -> dict:
    """Monte Carlo: rezultatas (dvimatis normalusis) ir žaidėjų taškai (Dirichlet dalys × komandos taškai) kartu.
    teams = {"home": {"c", "players": [{"id", "m"}]}, "away": {...}}; sąlygose v gali būti žaidėjo id. -> {"p", "se"}."""
    rho = max(-0.95, min(0.95, M.rho))
    alphas = {}
    for t in ("home", "away"):
        T = teams.get(t) or {"c": 1, "players": []}
        ms = [p["m"] for p in T["players"]]
        tot = sum(ms)
        scale = 0.98 / tot if tot > 0.98 else 1.0
        alphas[t] = ([p["id"] for p in T["players"]], [T["c"] * m * scale for m in ms] + [T["c"] * max(0.02, 1 - tot * scale)])
    for v, op, _ in conds:
        if op not in OPS or (v not in VARS and not any(v in ids for ids, _ in alphas.values())):
            raise ValueError(f"Netinkama sąlyga: {v} {op}")
    if np is not None:
        rng = np.random.default_rng(seed)
        z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
        H = np.maximum(0, np.rint(M.mh + M.sh * z1))
        A = np.maximum(0, np.rint(M.ma + M.sa * (rho * z1 + math.sqrt(1 - rho * rho) * z2)))
        P = {}
        for t, T in (("home", H), ("away", A)):
            ids, al = alphas[t]
            if ids:
                g = np.stack([rng.gamma(a, size=n) for a in al])
                g /= g.sum(axis=0)
                for j, pid in enumerate(ids):
                    P[pid] = np.rint(g[j] * T)
        get = lambda v: VARS[v](H, A) if v in VARS else P[v]  # noqa: E731
        mask = np.ones(n, dtype=bool)
        for v, op, x in conds:
            mask &= OPS[op](get(v), x)
        p = float(mask.mean())
    else:  # be NumPy (Brython): paprasti ciklai
        import random
        rnd = random.Random(seed)
        k = 0
        for _ in range(n):
            z1, z2 = rnd.gauss(0, 1), rnd.gauss(0, 1)
            h = max(0, round(M.mh + M.sh * z1))
            a = max(0, round(M.ma + M.sa * (rho * z1 + math.sqrt(1 - rho * rho) * z2)))
            vals = {"home": h, "away": a, "total": h + a, "diff": h - a}
            for t, T in (("home", h), ("away", a)):
                ids, al = alphas[t]
                if ids:
                    g = [rnd.gammavariate(x, 1) for x in al]
                    s = sum(g)
                    for j, pid in enumerate(ids):
                        vals[pid] = round(g[j] / s * T)
            if all(OPS[op](vals[v], x) for v, op, x in conds):
                k += 1
        p = k / n
    return {"p": p, "se": math.sqrt(p * (1 - p) / n)}


# ---------- CSV ir komandinė eilutė ----------
def read_rows(path: str) -> list[tuple[float, float, float]]:
    out = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for rec in csv.reader(f):
            nums = []
            for cell in rec:
                try:
                    nums.append(float(cell.replace(",", ".")))
                except ValueError:
                    pass
            if len(nums) >= 3:
                out.append((nums[0], nums[1], nums[2]))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Rungtynių tikimybės iš lažybų koeficientų")
    for k, name in (("hcp", "handikapas"), ("tot", "taškų suma"), ("home", "namų taškai"), ("away", "svečių taškai")):
        ap.add_argument(f"--{k}", metavar="CSV", help=f"{name}: linija, A, B")
    ap.add_argument("--home-name", default="Namai")
    ap.add_argument("--away-name", default="Svečiai")
    ap.add_argument("--cond", type=parse_cond, action="append", default=[],
                    help="sąlyga, pvz. home>75 (galima kartoti, visos sujungtos „ir“)")
    ap.add_argument("--top", type=int, default=10, help="kiek tikėtiniausių rezultatų rodyti")
    args = ap.parse_args(argv)

    F = {k: fit(k, read_rows(getattr(args, k))) if getattr(args, k) else None for k in ("hcp", "tot", "home", "away")}
    H, A = args.home_name, args.away_name
    if F["hcp"]:
        print(f"{H} laimi: {p_over(F['hcp'], 0) * 100:.2f}%   skirtumas {F['hcp'].mu:+.1f} (σ {F['hcp'].sigma:.1f})")
    if F["tot"]:
        print(f"Taškų suma: {F['tot'].mu:.1f} (σ {F['tot'].sigma:.1f})")
    M = score_model(F)
    if not M:
        print("Rezultato modeliui reikia bent dviejų rinkų (pvz. handikapo ir sumos arba abiejų komandų taškų).")
        return
    print(f"Tikėtinas rezultatas: {H} {M.mh:.1f} : {M.ma:.1f} {A}   σ {M.sh:.1f} / {M.sa:.1f}   ρ = {M.rho:.3f}   ({M.src})")
    if M.rho_tot is not None or M.rho_hcp is not None:
        parts = [f"pagal sumą {M.rho_tot:.3f}" if M.rho_tot is not None else "",
                 f"pagal handikapą {M.rho_hcp:.3f}" if M.rho_hcp is not None else ""]
        print("ρ atskirai: " + " · ".join(p for p in parts if p))
    cells = sorted((c for c in grid_cells(M) if c[0] != c[1]), key=lambda c: -c[2])[: args.top]
    print("\nTikėtiniausi rezultatai:")
    for i, (h, a, p) in enumerate(cells, 1):
        print(f"  {i:>2}. {h} : {a}   {p * 100:.2f}%")
    if args.cond:
        p = condition_probability(M, args.cond)
        conds = " ir ".join(f"{v} {op} {x:g}" for v, op, x in args.cond)
        print(f"\nP({conds}) = {p * 100:.2f}%   teisingas koeficientas {1 / p:.2f}" if p > 0 else f"\nP({conds}) = 0")


if __name__ == "__main__":
    main()
