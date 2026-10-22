/* Skaičiavimų modelis: koeficientai -> tikimybės, normaliojo skirstinio įvertis,
   rezultatų tinklelis ir sąlygų tikimybė. Be jokių puslapio (DOM) priklausomybių.
   Naršyklėje funkcijos tampa globaliomis, Node.js eksportuojamos per module.exports. */

// ---------- normalusis skirstinys ----------
function erf(x) { const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y; }
const Phi = z => 0.5 * (1 + erf(z / Math.SQRT2));
function PhiInv(p) { // Acklam
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const pl = 0.02425;
  if (p < pl) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1); }
  if (p > 1 - pl) { const q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1); }
  const q = p - 0.5, r = q * q;
  return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
}

// Each row -> threshold k and P(X > k) without margin. X = home−away margin for hcp, points otherwise.
function rowCalc(m, r) {
  const l = parseFloat(String(r.l).replace(",", ".")), a = parseFloat(String(r.a).replace(",", ".")), b = parseFloat(String(r.b).replace(",", "."));
  if (!isFinite(l) || !(a > 1) || !(b > 1)) return null;
  const ia = 1 / a, ib = 1 / b, book = ia + ib;
  return { k: m.key === "hcp" ? -l : l, pa: ia / book, pb: ib / book, margin: book - 1 };
}
function fit(m, rows) {
  const pts = rows.map(r => rowCalc(m, r)).filter(Boolean).map(c => ({ k: c.k, z: PhiInv(1 - Math.min(0.99, Math.max(0.01, c.pa))) }));
  if (!pts.length) return null;
  if (pts.length >= 2) {
    const n = pts.length, mk = pts.reduce((s, p) => s + p.k, 0) / n, mz = pts.reduce((s, p) => s + p.z, 0) / n;
    const sxx = pts.reduce((s, p) => s + (p.k - mk) ** 2, 0);
    if (sxx > 0) {
      const slope = pts.reduce((s, p) => s + (p.k - mk) * (p.z - mz), 0) / sxx;
      if (slope > 0.01) { const sigma = 1 / slope; return { mu: mk - mz * sigma, sigma, n, assumed: false }; }
    }
  }
  const sigma = m.sigma0;
  const mu = pts.reduce((s, p) => s + (p.k - p.z * sigma), 0) / pts.length;
  return { mu, sigma, n: pts.length, assumed: true };
}
const pOver = (f, k) => 1 - Phi((k - f.mu) / f.sigma);

// ---------- rezultato modelis ----------
// F = {hcp, tot, home, away}: kiekvienos rinkos fit() rezultatas arba null.
//
// Namų (H) ir svečių (A) taškai laikomi dvimačiu normaliuoju skirstiniu su μh, μa, σh, σa, ρ.
// Kiekviena rinka duoda vidurkį ir dispersiją, kurie tiesiškai priklauso nuo nežinomųjų:
//   handikapas  H − A:  μh − μa,   σh² + σa² − 2c        (c = ρ·σh·σa)
//   suma        H + A:  μh + μa,   σh² + σa² + 2c
//   namų taškai H:      μh,        σh²
//   svečių      A:      μa,        σa²
// Visos turimos rinkos sujungiamos svertiniu mažiausių kvadratų metodu (vidurkiai: svoris n/σ²,
// dispersijos: svoris n/σ⁴, t. y. pagal santykinę paklaidą). Kai duomenų nepakanka, labai silpnos
// prielaidos σh = σa ir ρ = 0 nulemia tik tai, ko rinkos neapibrėžia.
const SCORE_ROWS = {
  hcp: { m: [1, -1], v: [1, 1, -2], name: "handikapas" },
  tot: { m: [1, 1], v: [1, 1, 2], name: "suma" },
  home: { m: [1, 0], v: [1, 0, 0], name: "namų taškai" },
  away: { m: [0, 1], v: [0, 1, 0], name: "svečių taškai" },
};
function wls(rows) { // rows: [x[], y, w] -> mažiausių kvadratų sprendinys arba null
  const n = rows[0][0].length, A = Array.from({ length: n }, () => Array(n + 1).fill(0));
  for (const [x, y, w] of rows) for (let i = 0; i < n; i++) { for (let j = 0; j < n; j++) A[i][j] += w * x[i] * x[j]; A[i][n] += w * x[i] * y; }
  const scale = Math.max(...A.map((r, i) => Math.abs(r[i]))) || 1;
  for (let c = 0; c < n; c++) { // Gauso eliminacija su pagrindinio elemento parinkimu
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    if (Math.abs(A[p][c]) < 1e-12 * scale) return null;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = 0; r < n; r++) if (r !== c) { const f = A[r][c] / A[c][c]; for (let j = c; j <= n; j++) A[r][j] -= f * A[c][j]; }
  }
  return A.map((r, i) => r[n] / r[i]);
}
function scoreModel(F) {
  const keys = Object.keys(SCORE_ROWS).filter(k => F[k]);
  const mean = wls(keys.map(k => [SCORE_ROWS[k].m, F[k].mu, F[k].n / F[k].sigma ** 2]));
  if (!keys.length || !mean) return null;
  // Rinkos su numatyta σ (viena linija) beveik nesveria; prielaidos sveria dar mažiau.
  const vRows = keys.map(k => [SCORE_ROWS[k].v, F[k].sigma ** 2, (F[k].assumed ? 1e-3 : 1) * F[k].n / F[k].sigma ** 4]);
  const wMax = Math.max(...vRows.map(r => r[2]));
  vRows.push([[1, -1, 0], 0, 1e-6 * wMax], [[0, 0, 1], 0, 1e-6 * wMax]);
  let [vh, va, c] = wls(vRows);
  vh = Math.max(vh, 1); va = Math.max(va, 1);
  const sh = Math.sqrt(vh), sa = Math.sqrt(va), rho = Math.max(-0.95, Math.min(0.95, c / (sh * sa)));
  // Atskiri ρ įverčiai palyginimui: iš komandų taškų σ ir sumos arba handikapo σ.
  let rho_tot = null, rho_hcp = null;
  const real = k => F[k] && !F[k].assumed;
  if (real("home") && real("away")) {
    const vH = F.home.sigma ** 2, vA = F.away.sigma ** 2, d = 2 * Math.sqrt(vH * vA);
    if (real("tot")) rho_tot = (F.tot.sigma ** 2 - vH - vA) / d;
    if (real("hcp")) rho_hcp = (vH + vA - F.hcp.sigma ** 2) / d;
  }
  return { mh: mean[0], ma: mean[1], sh, sa, rho, rho_tot, rho_hcp, src: keys.map(k => SCORE_ROWS[k].name).join(" + ") };
}

// Visi sveiki rezultatai h : a per ±5σ su tikimybėmis (suma = 1).
function gridCells(M) {
  const rho = Math.max(-0.95, Math.min(0.95, M.rho)), cells = []; let tot = 0;
  for (let h = Math.max(0, Math.floor(M.mh - 5 * M.sh)); h <= Math.ceil(M.mh + 5 * M.sh); h++)
    for (let a = Math.max(0, Math.floor(M.ma - 5 * M.sa)); a <= Math.ceil(M.ma + 5 * M.sa); a++) {
      const x = (h - M.mh) / M.sh, y = (a - M.ma) / M.sa;
      const d = Math.exp(-(x * x - 2 * rho * x * y + y * y) / (2 * (1 - rho * rho)));
      cells.push([h, a, d]); tot += d;
    }
  cells.forEach(c => c[2] /= tot);
  return cells;
}

// ---------- sąlygos ----------
const VARS = [["home", h => h], ["away", (h, a) => a], ["total", (h, a) => h + a], ["diff", (h, a) => h - a]];
const OPS = { ">": (x, y) => x > y, ">=": (x, y) => x >= y, "<": (x, y) => x < y, "<=": (x, y) => x <= y, "=": (x, y) => x === y };

// conds = [{v: "home"|"away"|"total"|"diff", op: ">"|">="|"<"|"<="|"=", x: skaičius}], visos sujungtos „ir“.
function conditionProbability(M, conds) {
  const cs = conds.map(c => ({ f: VARS.find(v => v[0] === c.v)?.[1], op: OPS[c.op], x: Number(c.x) }));
  if (cs.some(c => !c.f || !c.op || !isFinite(c.x))) throw new Error("Netinkama sąlyga");
  let p = 0;
  for (const [h, a, w] of gridCells(M)) if (cs.every(c => c.op(c.f(h, a), c.x))) p += w;
  return p;
}

// ---------- CSV įkėlimas ----------
// Kiekvienoje eilutėje imami pirmi trys skaitiniai stulpeliai: linija, koeficientas A, koeficientas B
// (handikapui A = namai, B = svečiai; sumoms A = daugiau, B = mažiau).
// Jei yra stulpelis „market“/„rinka“, eilutės paskirstomos rinkoms pagal jo reikšmę.
// Skirtukas aptinkamas automatiškai (, ; arba tab); su ; ir tab leidžiamas dešimtainis kablelis.
const MARKET_ALIASES = [
  ["hcp", /^(hcp|handicap|handikap|handikapas|spread|fora)/i],
  ["tot", /^(tot|total|totals|suma|taškų suma|tasku suma|over|o\/u)/i],
  ["home", /^(home|namai|namų|namu|namų komandos|team1|1)/i],
  ["away", /^(away|svečiai|sveciai|svečių|veciu|team2|2)/i],
];
// Bendra CSV pagalba: eilutės, skirtukas (, ; arba tab), langelių skaidymas ir skaičių atpažinimas.
function csvReader(text) {
  const lines = String(text).replace(/^﻿/, "").split(/\r?\n/).filter(l => l.trim());
  const head = lines[0] || "";
  const delim = [["\t", (head.match(/\t/g) || []).length], [";", (head.match(/;/g) || []).length], [",", (head.match(/,/g) || []).length]]
    .sort((x, y) => y[1] - x[1])[0][0];
  const split = l => l.split(delim).map(c => c.trim().replace(/^"(.*)"$/, "$1").trim());
  const num = c => { const v = delim === "," ? c : c.replace(",", "."); return /^[-+]?\d+(\.\d+)?$/.test(v) ? Number(v) : NaN; };
  return { lines, split, num };
}
function parseOddsCSV(text) {
  const { lines, split, num } = csvReader(text);
  if (!lines.length) return { byMarket: {}, unassigned: [], count: 0 };
  const head = lines[0];
  let marketCol = -1, start = 0;
  const first = split(head);
  if (first.filter(c => isFinite(num(c))).length < 3) {
    start = 1;
    marketCol = first.findIndex(c => /^(market|rinka|turgus|tipas|type)$/i.test(c));
  }
  const byMarket = {}, unassigned = [];
  for (const line of lines.slice(start)) {
    const cells = split(line);
    const nums = cells.filter((c, i) => i !== marketCol).map(num).filter(Number.isFinite);
    if (nums.length < 3) continue;
    const row = { l: String(nums[0]), a: nums[1].toFixed(2), b: nums[2].toFixed(2) };
    const mv = marketCol >= 0 ? cells[marketCol] || "" : "";
    const key = mv && MARKET_ALIASES.find(([, re]) => re.test(mv))?.[0];
    if (key) (byMarket[key] ||= []).push(row); else unassigned.push(row);
  }
  const count = unassigned.length + Object.values(byMarket).reduce((s, r) => s + r.length, 0);
  return { byMarket, unassigned, count };
}

// ---------- žaidėjų koeficientai ----------
// Eilutė: žaidėjas, (rodiklis), linija, daugiau, mažiau. Žaidėjo vardas yra pirmas neskaitinis langelis
// (arba stulpelis „player“/„žaidėjas“), rodiklis imamas iš stulpelio „stat“/„rodiklis“, jei jis yra.
const STATS = [
  ["pts", "Taškai", /^(pts|points|taškai|taskai|taškų|tasku)/i],
  ["reb", "Atkovoti kamuoliai", /^(reb|rebounds|atkovot|atk)/i],
  ["ast", "Rez. perdavimai", /^(ast|assists|rez|perdav)/i],
  ["3pm", "Tritaškiai", /^(3pm|3pt|threes|tritašk|tritask)/i],
  ["pra", "Taškai + atk. + perd.", /^(pra|pts\s*\+\s*reb\s*\+\s*ast|t\s*\+\s*a\s*\+\s*p)/i],
  ["other", "Kita", /^(other|kita)/i],
];
const statKey = s => STATS.find(([, , re]) => re.test(String(s || "").trim()))?.[0] || null;
function parsePlayerCSV(text) {
  const { lines, split, num } = csvReader(text);
  if (!lines.length) return { rows: [], count: 0 };
  let nameCol = -1, statCol = -1, teamCol = -1, start = 0;
  const first = split(lines[0]);
  if (first.filter(c => isFinite(num(c))).length < 3) {
    start = 1;
    nameCol = first.findIndex(c => /^(player|žaidėjas|zaidejas|žaidėjai|vardas|name)$/i.test(c));
    statCol = first.findIndex(c => /^(stat|rodiklis|market|rinka|tipas|type)$/i.test(c));
    teamCol = first.findIndex(c => /^(team|komanda|side|pusė|puse)$/i.test(c));
  }
  const rows = [];
  for (const line of lines.slice(start)) {
    const cells = split(line);
    const skip = i => i === nameCol || i === statCol || i === teamCol;
    const nums = cells.filter((c, i) => !skip(i)).map(num).filter(Number.isFinite);
    if (nums.length < 3) continue;
    const name = nameCol >= 0 ? cells[nameCol] : cells.find((c, i) => !skip(i) && c && !isFinite(num(c)));
    if (!name) continue;
    rows.push({ n: name, s: statCol >= 0 ? statKey(cells[statCol]) : null, tm: teamCol >= 0 ? cells[teamCol] || "" : "",
      l: String(nums[0]), a: nums[1].toFixed(2), b: nums[2].toFixed(2) });
  }
  return { rows, count: rows.length };
}

// Žaidėjo tikėtini taškai iš vienos daugiau/mažiau linijos. Taškai laikomi normaliuoju skirstiniu, kurio
// dispersija proporcinga vidurkiui: X ~ N(μ, k·μ), k = 2.2 (pvz. μ 20 -> σ 6.6, μ 10 -> σ 4.7).
// Iš P(X > linija) = p:  linija − μ = z·√(k·μ), z = Φ⁻¹(1 − p)  ->  kvadratinė lygtis √μ atžvilgiu.
// Kai p arti 50 %, μ beveik lygus linijai ir k beveik nesvarbus.
const PLAYER_VAR_K = 2.2;
function playerMean(line, pOver, k = PLAYER_VAR_K) {
  const L = parseFloat(String(line).replace(",", "."));
  if (!(L > 0) || !(pOver > 0 && pOver < 1)) return null;
  const z = PhiInv(1 - Math.min(0.99, Math.max(0.01, pOver))), rk = Math.sqrt(k);
  const s = (-z * rk + Math.sqrt(z * z * k + 4 * L)) / 2;
  return s * s;
}

// ---------- Dirichlet žaidėjų dalių modelis ----------
// Komandos taškai T (iš tikslaus rezultato modelio) dalijami žaidėjams: dalys (S₁…S_k, S_kiti) ~ Dirichlet(c·m₁, …, c·m_k, c·m_kiti),
// žaidėjo taškai Xᵢ = Sᵢ·T. Vienam žaidėjui Sᵢ ~ Beta(c·mᵢ, c·(1−mᵢ)), nepriklausoma nuo T, todėl
// P(Xᵢ > linija) = E_T[1 − I_{linija/T}(c·mᵢ, c·(1−mᵢ))]. mᵢ parenkamas, kad ši tikimybė sutaptų su rinkos tikimybe be maržos;
// c (koncentracija) nustatoma ranka arba įvertinama iš žaidėjų, turinčių kelias linijas. Dalies sd = √(m(1−m)/(c+1)).
function lgamma(x) { // Lanczos
  const g = 7, k = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1; let a = k[0]; const t = x + g + 0.5;
  for (let i = 1; i < 9; i++) a += k[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
function betacf(a, b, x) { // grandininė trupmena nepilnajai beta funkcijai
  const TINY = 1e-300, qab = a + b, qap = a + 1, qam = a - 1;
  let c = 1, d = 1 - qab * x / qap; if (Math.abs(d) < TINY) d = TINY; d = 1 / d; let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < TINY) d = TINY; c = 1 + aa / c; if (Math.abs(c) < TINY) c = TINY; d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < TINY) d = TINY; c = 1 + aa / c; if (Math.abs(c) < TINY) c = TINY; d = 1 / d;
    const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-13) break;
  }
  return h;
}
function betaCDF(x, a, b) {
  if (x <= 0) return 0; if (x >= 1) return 1;
  const bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
}
const bisect = (f, lo, hi, it = 60) => { for (let i = 0; i < it; i++) { const mid = (lo + hi) / 2; if (f(mid)) hi = mid; else lo = mid; } return (lo + hi) / 2; };
const betaQuantile = (q, a, b) => bisect(x => betaCDF(x, a, b) >= q, 0, 1);
// Komandos taškų normalusis skirstinys kaip svoriai taškuose ±4σ.
function teamNodes(mu, sd, n = 41) {
  const out = []; let tot = 0;
  for (let i = 0; i < n; i++) { const z = -4 + 8 * i / (n - 1), t = mu + z * sd, w = Math.exp(-z * z / 2); if (t > 1) { out.push([t, w]); tot += w; } }
  return out.map(([t, w]) => [t, w / tot]);
}
function sharePOver(m, c, L, nodes) {
  const a = c * m, b = c * (1 - m); let p = 0;
  for (const [t, w] of nodes) p += w * (1 - betaCDF(L / t, a, b));
  return p;
}
const probit = p => PhiInv(Math.min(0.995, Math.max(0.005, p)));
const golden = (f, lo, hi, it = 60) => {
  const r = (Math.sqrt(5) - 1) / 2; let x1 = hi - r * (hi - lo), x2 = lo + r * (hi - lo), f1 = f(x1), f2 = f(x2);
  for (let i = 0; i < it; i++) {
    if (f1 < f2) { hi = x2; x2 = x1; f2 = f1; x1 = hi - r * (hi - lo); f1 = f(x1); }
    else { lo = x1; x1 = x2; f1 = f2; x2 = lo + r * (hi - lo); f2 = f(x2); }
  }
  return (lo + hi) / 2;
};
// lines = [{L, p}] (p = rinkos P(daugiau) be maržos). Grąžina vidutinę dalį m ir paklaidą probit skalėje.
function fitShare(lines, c, nodes) {
  const sse = m => lines.reduce((s, l) => s + (probit(sharePOver(m, c, l.L, nodes)) - probit(l.p)) ** 2, 0);
  const m = Math.exp(golden(u => sse(Math.exp(u)), Math.log(0.002), Math.log(0.9)));
  return { m, sse: sse(m) };
}
const pointsQuantile = (q, m, c, nodes, hi) => {
  const a = c * m, b = c * (1 - m);
  return bisect(x => nodes.reduce((s, [t, w]) => s + w * betaCDF(x / t, a, b), 0) >= q, 0, hi, 40);
};
// team = {mu, sd}; players = [{id, n, lines: [{L, p}]}]. Grąžina kiekvieno žaidėjo dalį, intervalus ir modelio tikimybes.
function dirichletTeam(team, players, c) {
  const nodes = teamNodes(team.mu, team.sd);
  let sse = 0;
  const res = players.map(pl => {
    const f = fitShare(pl.lines, c, nodes); sse += f.sse;
    const m = f.m, a = c * m, b = c * (1 - m);
    return { id: pl.id, n: pl.n, m, lo: betaQuantile(0.1, a, b), hi: betaQuantile(0.9, a, b), mean: m * team.mu,
      ptsLo: pointsQuantile(0.1, m, c, nodes, team.mu + 6 * team.sd), ptsHi: pointsQuantile(0.9, m, c, nodes, team.mu + 6 * team.sd),
      lines: pl.lines.map(l => ({ L: l.L, p: l.p, pm: sharePOver(m, c, l.L, nodes) })) };
  });
  const sum = res.reduce((s, r) => s + r.m, 0);
  return { c, mu: team.mu, sd: team.sd, players: res, rest: 1 - sum, sse };
}
// Koncentracijos įvertis: c, kuriam modelis geriausiai atkuria visas žaidėjų linijas (informatyvūs tik žaidėjai su ≥ 2 linijomis).
// groups = [{team: {mu, sd}, players: [...]}]. Grąžina null, jei nėra žaidėjo su keliomis linijomis.
function fitConcentration(groups) {
  const multi = groups.flatMap(g => g.players.filter(p => p.lines.length >= 2).map(p => ({ nodes: teamNodes(g.team.mu, g.team.sd), lines: p.lines })));
  if (!multi.length) return null;
  const obj = c => multi.reduce((s, p) => s + fitShare(p.lines, c, p.nodes).sse, 0);
  let best = Math.log(3), bv = Infinity;
  for (let i = 0; i <= 24; i++) { const u = Math.log(3) + (Math.log(1500) - Math.log(3)) * i / 24, v = obj(Math.exp(u)); if (v < bv) { bv = v; best = u; } }
  const step = (Math.log(1500) - Math.log(3)) / 24;
  return { c: Math.exp(golden(u => obj(Math.exp(u)), best - step, best + step, 30)), players: multi.length };
}

// ---------- Monte Carlo: rezultatas ir žaidėjų taškai kartu ----------
function mulberry32(seed) { return () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function gammaSample(k, rnd, norm) { // Marsaglia–Tsang
  if (k < 1) return gammaSample(k + 1, rnd, norm) * Math.pow(rnd(), 1 / k);
  const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) { let x, v; do { x = norm(); v = 1 + c * x; } while (v <= 0); v = v * v * v; const u = rnd();
    if (u < 1 - 0.0331 * x * x * x * x || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v; }
}
// M = scoreModel rezultatas; teams = {home: {c, players: [{id, m}]}, away: {...}}. Grąžina n imčių: H, A ir kiekvieno žaidėjo taškai.
function simulateGame(M, teams, n = 40000, seed = 12345) {
  const rnd = mulberry32(seed); let spare = null;
  const norm = () => { if (spare !== null) { const s = spare; spare = null; return s; }
    let u = 0; while (u === 0) u = rnd(); const r = Math.sqrt(-2 * Math.log(u)), th = 2 * Math.PI * rnd(); spare = r * Math.sin(th); return r * Math.cos(th); };
  const rho = Math.max(-0.95, Math.min(0.95, M.rho)), H = new Float64Array(n), A = new Float64Array(n), P = {};
  const tm = ["home", "away"].map(t => { const T = teams[t] || { c: 1, players: [] };
    const ms = T.players.map(p => p.m), sum = ms.reduce((s, m) => s + m, 0), scale = sum > 0.98 ? 0.98 / sum : 1;
    T.players.forEach(p => { P[p.id] = new Float64Array(n); });
    return { c: T.c, ids: T.players.map(p => p.id), alpha: [...ms.map(m => T.c * m * scale), T.c * Math.max(0.02, 1 - sum * scale)] }; });
  for (let i = 0; i < n; i++) {
    const z1 = norm(), z2 = norm();
    H[i] = Math.max(0, Math.round(M.mh + M.sh * z1));
    A[i] = Math.max(0, Math.round(M.ma + M.sa * (rho * z1 + Math.sqrt(1 - rho * rho) * z2)));
    tm.forEach((T, ti) => {
      if (!T.ids.length) return;
      const g = T.alpha.map(a => gammaSample(a, rnd, norm)), tot = g.reduce((s, x) => s + x, 0), pts = ti ? A[i] : H[i];
      T.ids.forEach((id, j) => { P[id][i] = Math.round(g[j] / tot * pts); });
    });
  }
  return { n, H, A, P };
}
// Sąlygos kaip conditionProbability, papildomai v = žaidėjo id (jo taškai). Grąžina {p, se}.
function conditionProbabilityMC(sim, conds) {
  const get = v => v === "home" ? i => sim.H[i] : v === "away" ? i => sim.A[i] : v === "total" ? i => sim.H[i] + sim.A[i]
    : v === "diff" ? i => sim.H[i] - sim.A[i] : sim.P[v] ? (arr => i => arr[i])(sim.P[v]) : null;
  const cs = conds.map(c => ({ f: get(c.v), op: OPS[c.op], x: Number(c.x) }));
  if (cs.some(c => !c.f || !c.op || !isFinite(c.x))) throw new Error("Netinkama sąlyga");
  let k = 0;
  for (let i = 0; i < sim.n; i++) if (cs.every(c => c.op(c.f(i), c.x))) k++;
  const p = k / sim.n;
  return { p, se: Math.sqrt(p * (1 - p) / sim.n) };
}

// Komanda iš teksto: „namai“/„svečiai“ (home/away) arba komandos pavadinimas; null, jei neaišku.
const normTxt = s => String(s || "").trim().toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");
function teamKey(s, home, away) {
  const v = normTxt(s);
  if (!v) return null;
  if (/^(home|namai|namu|namu komanda|team ?1|1|h)$/.test(v)) return "home";
  if (/^(away|sveciai|sveciu|sveciu komanda|team ?2|2|a)$/.test(v)) return "away";
  const h = normTxt(home), a = normTxt(away);
  const hit = t => t && (t === v || t.includes(v) || v.includes(t));
  if (hit(h) && !hit(a)) return "home";
  if (hit(a) && !hit(h)) return "away";
  return null;
}

if (typeof module !== "undefined") {
  module.exports = { erf, Phi, PhiInv, rowCalc, fit, pOver, scoreModel, gridCells, conditionProbability, VARS, OPS, parseOddsCSV, STATS, statKey, parsePlayerCSV, teamKey, playerMean, PLAYER_VAR_K,
    betaCDF, betaQuantile, teamNodes, sharePOver, fitShare, dirichletTeam, fitConcentration, simulateGame, conditionProbabilityMC };
}
