// Skaičiavimų patikra: node tests/model.test.js
const assert = require("node:assert/strict");
const M = require("../app/model.js");

const HCP = { key: "hcp", sigma0: 12 };
const TOT = { key: "tot", sigma0: 16 };
const rows = list => list.map(([l, a, b]) => ({ l, a, b }));

// Fenerbahce – Partizan, Betsson handikapai
const hcp = rows([[-14.5, 2.45, 1.50], [-13.5, 2.30, 1.62], [-12.5, 2.15, 1.70], [-11.5, 1.95, 1.85],
                  [-10.5, 1.85, 1.95], [-9.5, 1.70, 2.15], [-8.5, 1.62, 2.30], [-7.5, 1.50, 2.45]]);
const tot = rows([[160.5, 1.55, 2.35], [163.5, 1.75, 2.05], [165.5, 1.90, 1.90], [167.5, 2.05, 1.75], [170.5, 2.35, 1.55]]);

const near = (x, y, eps, msg) => assert.ok(Math.abs(x - y) < eps, `${msg}: ${x} != ${y}`);

// Normalusis skirstinys
near(M.Phi(1.96), 0.975, 1e-4, "Phi(1.96)");
near(M.PhiInv(0.975), 1.96, 1e-3, "PhiInv(0.975)");

// Koeficientai -> tikimybės be maržos
const r = M.rowCalc(HCP, hcp[3]);
near(r.pa, 0.4868, 1e-4, "P(namai -11.5)");
near(r.margin, 0.0534, 1e-4, "marža");

// Normaliojo skirstinio pritaikymas
const F = { hcp: M.fit(HCP, hcp), tot: M.fit(TOT, tot), home: null, away: null };
near(F.hcp.mu, 11.0, 0.05, "vid. skirtumas");
near(F.hcp.sigma, 11.31, 0.05, "σ skirtumo");
near(M.pOver(F.hcp, 0), 0.8346, 1e-3, "P(Fenerbahce laimi)");
near(F.tot.mu, 165.5, 0.05, "vid. suma");

// Rezultato modelis ir tinklelis
const SM = M.scoreModel(F);
near(SM.mh, 88.25, 0.05, "namų vidurkis");
near(SM.ma, 77.25, 0.05, "svečių vidurkis");
near(SM.rho, 0.491, 0.01, "koreliacija");
const cells = M.gridCells(SM);
near(cells.reduce((s, c) => s + c[2], 0), 1, 1e-9, "tinklelio suma");

// Bendras modelis iš visų keturių rinkų: koeficientai sugeneruoti iš žinomų parametrų
// (μh 88, μa 77, σh 11, σa 9, ρ 0.3) su 5 % marža; modelis turi juos atkurti.
const TRUE = { mh: 88, ma: 77, sh: 11, sa: 9, rho: 0.3 };
const cov = TRUE.rho * TRUE.sh * TRUE.sa;
const gen = (key, mu, sd, lines) => lines.map(l => {
  const k = key === "hcp" ? -l : l, p = 1 - M.Phi((k - mu) / sd);
  return { l, a: 1 / (p * 1.025), b: 1 / ((1 - p) * 1.025) };
});
const mk = key => ({ key, sigma0: { hcp: 12, tot: 16, home: 10, away: 10 }[key] });
const G = {
  hcp: M.fit(mk("hcp"), gen("hcp", TRUE.mh - TRUE.ma, Math.sqrt(TRUE.sh ** 2 + TRUE.sa ** 2 - 2 * cov), [-16.5, -13.5, -10.5, -7.5, -4.5])),
  tot: M.fit(mk("tot"), gen("tot", TRUE.mh + TRUE.ma, Math.sqrt(TRUE.sh ** 2 + TRUE.sa ** 2 + 2 * cov), [157.5, 162.5, 165.5, 168.5, 173.5])),
  home: M.fit(mk("home"), gen("home", TRUE.mh, TRUE.sh, [82.5, 86.5, 88.5, 90.5, 94.5])),
  away: M.fit(mk("away"), gen("away", TRUE.ma, TRUE.sa, [72.5, 75.5, 77.5, 79.5, 82.5])),
};
const GM = M.scoreModel(G);
near(GM.mh, 88, 0.05, "bendras: namų vidurkis"); near(GM.ma, 77, 0.05, "bendras: svečių vidurkis");
near(GM.sh, 11, 0.05, "bendras: σ namų"); near(GM.sa, 9, 0.05, "bendras: σ svečių");
near(GM.rho, 0.3, 0.005, "bendras: ρ"); near(GM.rho_tot, 0.3, 0.005, "ρ pagal sumą"); near(GM.rho_hcp, 0.3, 0.005, "ρ pagal handikapą");
assert.equal(GM.src, "handikapas + suma + namų taškai + svečių taškai");
// Be bendrų rinkų koreliacija neįvertinama (0), be komandų rinkų σ laikomos lygiomis.
const onlyTeams = M.scoreModel({ home: G.home, away: G.away });
near(onlyTeams.rho, 0, 1e-6, "tik komandų taškai: ρ = 0"); near(onlyTeams.sh, 11, 0.05, "tik komandų taškai: σ namų");
const noTeams = M.scoreModel({ hcp: G.hcp, tot: G.tot });
near(noTeams.sh, noTeams.sa, 1e-6, "be komandų rinkų σ vienodos");
// Suma + vienos komandos taškai: likusi komanda ir ρ įvertinami iš likusių rinkų.
const three = M.scoreModel({ hcp: G.hcp, tot: G.tot, home: G.home });
near(three.sa, 9, 0.05, "be svečių rinkos: σ svečių"); near(three.rho, 0.3, 0.005, "be svečių rinkos: ρ");
assert.equal(M.scoreModel({ hcp: G.hcp }), null, "vien handikapo nepakanka");

// Sąlygos
const pAway = M.conditionProbability(SM, [{ v: "diff", op: "<", x: 0 }]);
const pHome = M.conditionProbability(SM, [{ v: "diff", op: ">", x: 0 }]);
const pDraw = M.conditionProbability(SM, [{ v: "diff", op: "=", x: 0 }]);
near(pAway + pHome + pDraw, 1, 1e-9, "laimi + pralaimi + lygiosios");
const pBoth = M.conditionProbability(SM, [{ v: "home", op: ">", x: 75 }, { v: "away", op: ">", x: 75 }, { v: "diff", op: "<", x: 0 }]);
assert.ok(pBoth < pAway, "papildomos sąlygos mažina tikimybę");
near(pBoth, 0.1018, 2e-3, "abi > 75 ir laimi Partizan");
assert.throws(() => M.conditionProbability(SM, [{ v: "nėra", op: ">", x: 1 }]));

// CSV įkėlimas
const fs = require("node:fs");
const path = require("node:path");
const own = M.parseOddsCSV(fs.readFileSync(path.join(__dirname, "../data/fenerbahce_partizan_betsson_handicap.csv"), "utf8"));
assert.equal(own.count, 8, "esamas handikapo failas");
assert.deepEqual(own.byMarket.hcp?.[3], { l: "-11.5", a: "1.95", b: "1.85" }, "market=Handicap -> hcp");

const semi = M.parseOddsCSV("Linija;Daugiau;Mažiau\n165,5;1,90;1,90\n167,5;2,05;1,75\n");
assert.equal(semi.unassigned.length, 2, "kabliataškis ir dešimtainis kablelis");
assert.deepEqual(semi.unassigned[0], { l: "165.5", a: "1.90", b: "1.90" });

const combo = M.parseOddsCSV(["rinka,linija,a,b", "handikapas,-11.5,1.95,1.85", "taškų suma,165.5,1.90,1.90",
  "namai,88.5,1.85,1.95", "svečiai,77.5,1.90,1.90", "tekstas be skaičių"].join("\n"));
assert.equal(combo.byMarket.hcp.length, 1); assert.equal(combo.byMarket.tot.length, 1);
assert.equal(combo.byMarket.home.length, 1); assert.equal(combo.byMarket.away.length, 1);
assert.equal(combo.count, 4, "eilutės be skaičių praleidžiamos");

// Žaidėjų koeficientai
const pl = M.parsePlayerCSV(["Žaidėjas;Linija;Daugiau;Mažiau", "Tyrese Maxey;22,5;1,80;1,86", "Joel Embiid;22,5;1,90;1,76",
  "VJ Edgecombe;11,5;1,71;1,95"].join("\n"));
assert.equal(pl.count, 3, "žaidėjai: kabliataškis");
const plStat = M.parsePlayerCSV("player,stat,line,over,under\nLeBron James,Rebounds,7.5,1.85,1.85\nJaylen Brown,taškai,20.5,1.86,1.80\n");
assert.equal(plStat.rows[0].s, "reb"); assert.equal(plStat.rows[1].s, "pts");
const plNoHead = M.parsePlayerCSV("LeBron James,15.5,1.86,1.80\nbe skaičių\n");
assert.deepEqual(plNoHead.rows, [{ n: "LeBron James", s: null, tm: "", l: "15.5", a: "1.86", b: "1.80" }], "be antraštės");
const maxey = M.rowCalc(TOT, pl.rows[0]);
near(maxey.pa, (1 / 1.80) / (1 / 1.80 + 1 / 1.86), 1e-12, "Maxey daugiau be maržos");

assert.deepEqual(pl.rows[0], { n: "Tyrese Maxey", s: null, tm: "", l: "22.5", a: "1.80", b: "1.86" }, "be komandos stulpelio");
const plTeam = M.parsePlayerCSV("komanda,žaidėjas,linija,daugiau,mažiau\nPhiladelphia 76ers,Tyrese Maxey,22.5,1.80,1.86\nsvečiai,Jaylen Brown,20.5,1.86,1.80\n");
assert.equal(plTeam.rows[0].tm, "Philadelphia 76ers");
assert.equal(M.teamKey(plTeam.rows[0].tm, "76ers", "Celtics"), "home", "komanda pagal pavadinimą");
assert.equal(M.teamKey(plTeam.rows[1].tm, "76ers", "Celtics"), "away", "svečiai");
assert.equal(M.teamKey("Namų", "", ""), "home", "su lietuviškomis raidėmis");
assert.equal(M.teamKey("Lakers", "76ers", "Celtics"), null, "nežinoma komanda");
assert.equal(M.teamKey("Partizan", "", "Partizan"), "away", "tuščias namų pavadinimas netrukdo");

// Žaidėjo tikėtini taškai: atkuriamas žinomas vidurkis iš sugeneruotos tikimybės
near(M.playerMean(22.5, 0.5), 22.5, 1e-9, "p = 50 % -> vidurkis = linija");
for (const mu of [12, 14.5, 17, 19]) {
  const p = 1 - M.Phi((15.5 - mu) / Math.sqrt(M.PLAYER_VAR_K * mu));
  near(M.playerMean(15.5, p), mu, 0.05, `žaidėjo vidurkis ${mu}`);
}
assert.ok(M.playerMean(22.5, 0.55) > 22.5 && M.playerMean(22.5, 0.45) < 22.5, "daugiau tikėtiniau -> vidurkis virš linijos");
assert.equal(M.playerMean("", 0.5), null);

// Dirichlet: beta skirstinys (palyginta su scipy)
near(M.betaCDF(0.3, 2, 5), 0.579825, 1e-6, "beta CDF");
near(M.betaQuantile(0.9, 12, 48), 0.2678191, 1e-6, "beta kvantilis");
// Vienos linijos žaidėjas: modelis tiksliai atkuria rinkos tikimybę
const TEAM = { mu: 112, sd: 11 }, NODES = M.teamNodes(TEAM.mu, TEAM.sd);
const one = M.fitShare([{ L: 22.5, p: 0.508 }], 60, NODES);
near(M.sharePOver(one.m, 60, 22.5, NODES), 0.508, 1e-4, "viena linija atkuriama");
// Žinomi parametrai (m 0.2, c 60) -> kelių linijų tikimybės -> modelis atkuria m ir c
const TRUE_C = 60, TRUE_M = 0.2, LINES = [17.5, 20.5, 22.5, 24.5, 27.5];
const gl = LINES.map(L => ({ L, p: M.sharePOver(TRUE_M, TRUE_C, L, NODES) }));
near(M.fitShare(gl, TRUE_C, NODES).m, TRUE_M, 1e-4, "dalis iš kelių linijų");
const fc = M.fitConcentration([{ team: TEAM, players: [{ id: "x", n: "X", lines: gl }] }]);
near(fc.c, TRUE_C, 0.5, "koncentracija iš kelių linijų");
assert.equal(M.fitConcentration([{ team: TEAM, players: [{ id: "x", n: "X", lines: [gl[2]] }] }]), null, "viena linija c neįvertina");
// Komanda: dalys, likutis, intervalai
const DT = M.dirichletTeam(TEAM, [{ id: "a", n: "A", lines: [{ L: 22.5, p: 0.508 }] }, { id: "b", n: "B", lines: [{ L: 11.5, p: 0.533 }] }], 60);
near(DT.rest, 1 - DT.players[0].m - DT.players[1].m, 1e-12, "likusi dalis");
assert.ok(DT.players[0].lo < DT.players[0].m && DT.players[0].m < DT.players[0].hi, "dalies intervalas");
near(DT.players[0].mean, DT.players[0].m * TEAM.mu, 1e-9, "tikėtini taškai = dalis × komandos taškai");
// Monte Carlo sutampa su tiksliais skaičiavimais
const SMd = { mh: 112, ma: 108, sh: 11, sa: 11, rho: 0.1 };
const sim = M.simulateGame(SMd, { home: { c: 60, players: DT.players.map(p => ({ id: p.id, m: p.m })) }, away: { c: 60, players: [] } }, 40000, 1);
const mean = arr => arr.reduce((x, y) => x + y, 0) / arr.length;
near(mean(sim.P.a), DT.players[0].m * 112, 0.25, "MC žaidėjo vidurkis");
near(M.conditionProbabilityMC(sim, [{ v: "a", op: ">", x: 22.5 }]).p, DT.players[0].lines[0].pm, 0.012, "MC P(daugiau) ≈ modelis");
near(M.conditionProbabilityMC(sim, [{ v: "diff", op: ">", x: 0 }]).p, M.conditionProbability(SMd, [{ v: "diff", op: ">", x: 0 }]), 0.01, "MC rezultatas ≈ tinklelis");
const both = M.conditionProbabilityMC(sim, [{ v: "a", op: ">", x: 22.5 }, { v: "b", op: ">", x: 11.5 }]).p;
assert.ok(both < 0.6 && both > 0.15, "dviejų žaidėjų sąlyga");
assert.throws(() => M.conditionProbabilityMC(sim, [{ v: "nera", op: ">", x: 1 }]));

console.log("Visi skaičiavimų testai praėjo.");
