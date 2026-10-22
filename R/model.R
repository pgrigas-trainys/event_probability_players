# Skaičiavimų modelis R kalba (tas pats kaip app/model.js ir python/model.py).
#
# Koeficientai -> tikimybės be maržos, normaliojo skirstinio įvertis kiekvienai rinkai,
# abiejų komandų taškų modelis su koreliacija, rezultatų tinklelis ir sąlygų tikimybė.
# Naudojamas tik bazinis R.
#
# Paleidimas iš repozitorijos šaknies:
#   Rscript R/model.R --hcp data/hcp.csv --tot data/tot.csv \
#     --home-name Fenerbahce --away-name Partizan \
#     --cond "home>75" --cond "away>75" --cond "diff<0"
#
# Naudojimas iš kito R kodo:  source("R/model.R")
#
# CSV failuose naudojami pirmi trys skaitiniai stulpeliai: linija, koeficientas A, koeficientas B.
# Handikapui linija yra namų komandos handikapas (pvz. -11.5), A = namai, B = svečiai.
# Sumoms ir komandų taškams A = daugiau, B = mažiau.

SIGMA0 <- c(hcp = 12, tot = 16, home = 10, away = 10)

# ---------- koeficientai -> tikimybės ----------
row_calc <- function(market, line, a, b) {
  if (!is.finite(line) || !(a > 1 && b > 1)) return(NULL)
  ia <- 1 / a; ib <- 1 / b; book <- ia + ib
  list(k = if (market == "hcp") -line else line,   # slenkstis: X > k
       pa = ia / book, pb = ib / book, margin = book - 1)
}

# ---------- normaliojo skirstinio įvertis ----------
# P(X > k) = 1 - Φ((k - μ)/σ): z = Φ⁻¹(1 - p) tiesiškai priklauso nuo k, nuolydis = 1/σ.
fit_market <- function(market, rows) {
  calc <- Filter(Negate(is.null), lapply(seq_len(nrow(rows)), function(i)
    row_calc(market, rows[i, 1], rows[i, 2], rows[i, 3])))
  if (!length(calc)) return(NULL)
  k <- vapply(calc, `[[`, 0, "k")
  z <- qnorm(1 - pmin(0.99, pmax(0.01, vapply(calc, `[[`, 0, "pa"))))
  n <- length(k)
  if (n >= 2) {
    sxx <- sum((k - mean(k))^2)
    if (sxx > 0) {
      slope <- sum((k - mean(k)) * (z - mean(z))) / sxx
      if (slope > 0.01) {
        sigma <- 1 / slope
        return(list(mu = mean(k) - mean(z) * sigma, sigma = sigma, n = n, assumed = FALSE))
      }
    }
  }
  sigma <- SIGMA0[[market]]
  list(mu = mean(k - z * sigma), sigma = sigma, n = n, assumed = TRUE)
}

p_over <- function(f, k) 1 - pnorm((k - f$mu) / f$sigma)

# ---------- rezultato modelis ----------
# Namų (H) ir svečių (A) taškai – dvimatis normalusis skirstinys su μh, μa, σh, σa, ρ.
# Kiekviena rinka duoda vidurkį ir dispersiją, tiesiškai priklausančius nuo nežinomųjų:
#   handikapas  H − A:  μh − μa,   σh² + σa² − 2c        (c = ρ·σh·σa)
#   suma        H + A:  μh + μa,   σh² + σa² + 2c
#   namų taškai H:      μh,        σh²
#   svečių      A:      μa,        σa²
# Visos turimos rinkos sujungiamos svertiniu mažiausių kvadratų metodu (vidurkiai: svoris n/σ²,
# dispersijos: n/σ⁴). Labai silpnos prielaidos σh = σa ir ρ = 0 lemia tik tai, ko rinkos neapibrėžia.
SCORE_ROWS <- list(
  hcp  = list(m = c(1, -1), v = c(1, 1, -2), name = "handikapas"),
  tot  = list(m = c(1, 1),  v = c(1, 1, 2),  name = "suma"),
  home = list(m = c(1, 0),  v = c(1, 0, 0),  name = "namų taškai"),
  away = list(m = c(0, 1),  v = c(0, 1, 0),  name = "svečių taškai"))

.wls <- function(X, y, w) {
  A <- crossprod(X, w * X)
  if (rcond(A) < 1e-12) return(NULL)
  as.vector(solve(A, crossprod(X, w * y)))
}

score_model <- function(F) {
  keys <- Filter(function(k) !is.null(F[[k]]), names(SCORE_ROWS))
  if (!length(keys)) return(NULL)
  num <- function(k, f) as.numeric(F[[k]][[f]])
  sig2 <- vapply(keys, function(k) num(k, "sigma")^2, 0)
  n <- vapply(keys, function(k) num(k, "n"), 0)
  assumed <- vapply(keys, function(k) isTRUE(F[[k]]$assumed), TRUE)
  Xm <- do.call(rbind, lapply(keys, function(k) SCORE_ROWS[[k]]$m))
  mean <- .wls(Xm, vapply(keys, function(k) num(k, "mu"), 0), n / sig2)
  if (is.null(mean)) return(NULL)
  wv <- ifelse(assumed, 1e-3, 1) * n / sig2^2
  Xv <- rbind(do.call(rbind, lapply(keys, function(k) SCORE_ROWS[[k]]$v)), c(1, -1, 0), c(0, 0, 1))
  v <- .wls(Xv, c(sig2, 0, 0), c(wv, 1e-6 * max(wv), 1e-6 * max(wv)))
  sh <- sqrt(max(v[1], 1)); sa <- sqrt(max(v[2], 1))
  rho <- max(-0.95, min(0.95, v[3] / (sh * sa)))
  real <- function(k) !is.null(F[[k]]) && !isTRUE(F[[k]]$assumed)
  rho_tot <- NA_real_; rho_hcp <- NA_real_
  if (real("home") && real("away")) {
    vH <- num("home", "sigma")^2; vA <- num("away", "sigma")^2; d <- 2 * sqrt(vH * vA)
    if (real("tot")) rho_tot <- (num("tot", "sigma")^2 - vH - vA) / d
    if (real("hcp")) rho_hcp <- (vH + vA - num("hcp", "sigma")^2) / d
  }
  list(mh = mean[1], ma = mean[2], sh = sh, sa = sa, rho = rho, rho_tot = rho_tot, rho_hcp = rho_hcp,
       src = paste(vapply(keys, function(k) SCORE_ROWS[[k]]$name, ""), collapse = " + "))
}

# Visi sveiki rezultatai h : a per ±5σ su dvimačio normaliojo tankiu, sunormuoti iki 1.
grid_cells <- function(M) {
  rho <- max(-0.95, min(0.95, M$rho))
  h <- seq(max(0, floor(M$mh - 5 * M$sh)), ceiling(M$mh + 5 * M$sh))
  a <- seq(max(0, floor(M$ma - 5 * M$sa)), ceiling(M$ma + 5 * M$sa))
  g <- expand.grid(h = h, a = a)
  x <- (g$h - M$mh) / M$sh; y <- (g$a - M$ma) / M$sa
  d <- exp(-(x * x - 2 * rho * x * y + y * y) / (2 * (1 - rho * rho)))
  g$p <- d / sum(d)
  g
}

# ---------- sąlygos ----------
VARS <- list(home = function(h, a) h, away = function(h, a) a,
             total = function(h, a) h + a, diff = function(h, a) h - a)
OPS <- list(">" = `>`, ">=" = `>=`, "<" = `<`, "<=" = `<=`, "=" = `==`)

# conds: sąrašas list(v = "home", op = ">", x = 75), visos sujungtos „ir“.
condition_probability <- function(M, conds) {
  g <- grid_cells(M)
  ok <- rep(TRUE, nrow(g))
  for (c in conds) {
    if (is.null(VARS[[c$v]]) || is.null(OPS[[c$op]])) stop("Netinkama sąlyga: ", c$v, " ", c$op)
    ok <- ok & OPS[[c$op]](VARS[[c$v]](g$h, g$a), c$x)
  }
  sum(g$p[ok])
}

parse_cond <- function(text) {
  m <- regmatches(text, regexec("^\\s*(home|away|total|diff)\\s*(>=|<=|>|<|=)\\s*(-?[0-9]+(?:[.,][0-9]+)?)\\s*$", text))[[1]]
  if (!length(m)) stop("Sąlyga turi būti pvz. home>75, diff<0: ", text)
  list(v = m[2], op = m[3], x = as.numeric(sub(",", ".", m[4])))
}

# ---------- Dirichlet žaidėjų dalių modelis ----------
# Komandos taškai T (iš rezultato modelio) dalijami žaidėjams: dalys (S₁…S_k, S_kiti) ~ Dirichlet(c·m₁, …, c·m_k, c·m_kiti),
# žaidėjo taškai Xᵢ = Sᵢ·T. Vienam žaidėjui Sᵢ ~ Beta(c·mᵢ, c·(1−mᵢ)), nepriklausoma nuo T, todėl
# P(Xᵢ > linija) = E_T[1 − pbeta(linija/T, c·mᵢ, c·(1−mᵢ))]. mᵢ parenkamas, kad ši tikimybė atitiktų rinką be maržos;
# c (koncentracija) nustatoma ranka arba įvertinama iš žaidėjų su keliomis linijomis. Tas pats kaip app/model.js.
team_nodes <- function(mu, sd, n = 41) {
  z <- -4 + 8 * (seq_len(n) - 1) / (n - 1)
  t <- mu + z * sd; w <- exp(-z * z / 2)
  keep <- t > 1
  list(t = t[keep], w = w[keep] / sum(w[keep]))
}
share_p_over <- function(m, c, L, nodes) sum(nodes$w * (1 - pbeta(L / nodes$t, c * m, c * (1 - m))))
.probit <- function(p) qnorm(pmin(0.995, pmax(0.005, p)))
.golden <- function(f, lo, hi, it = 60) {
  r <- (sqrt(5) - 1) / 2
  x1 <- hi - r * (hi - lo); x2 <- lo + r * (hi - lo); f1 <- f(x1); f2 <- f(x2)
  for (i in seq_len(it)) {
    if (f1 < f2) { hi <- x2; x2 <- x1; f2 <- f1; x1 <- hi - r * (hi - lo); f1 <- f(x1) }
    else { lo <- x1; x1 <- x2; f1 <- f2; x2 <- lo + r * (hi - lo); f2 <- f(x2) }
  }
  (lo + hi) / 2
}
.bisect <- function(f, lo, hi, it = 60) {
  for (i in seq_len(it)) { mid <- (lo + hi) / 2; if (f(mid)) hi <- mid else lo <- mid }
  (lo + hi) / 2
}
# lines: sąrašas list(L = linija, p = rinkos P(daugiau)). Grąžina vidutinę dalį m ir paklaidą probit skalėje.
fit_share <- function(lines, c, nodes) {
  L <- vapply(lines, function(l) as.numeric(l$L), 0); p <- vapply(lines, function(l) as.numeric(l$p), 0)
  sse <- function(m) sum((.probit(vapply(L, function(x) share_p_over(m, c, x, nodes), 0)) - .probit(p))^2)
  m <- exp(.golden(function(u) sse(exp(u)), log(0.002), log(0.9), 45))
  list(m = m, sse = sse(m))
}
.points_quantile <- function(q, m, c, nodes, hi)
  .bisect(function(x) sum(nodes$w * pbeta(x / nodes$t, c * m, c * (1 - m))) >= q, 0, hi, 40)

# team = list(mu, sd); players = sąrašas list(id, n, lines = list(list(L, p), ...)).
dirichlet_team <- function(team, players, c) {
  mu <- as.numeric(team$mu); sd <- as.numeric(team$sd); c <- as.numeric(c)
  nodes <- team_nodes(mu, sd); hi <- mu + 6 * sd; sse <- 0
  res <- lapply(players, function(pl) {
    f <- fit_share(pl$lines, c, nodes); sse <<- sse + f$sse
    m <- f$m; a <- c * m; b <- c * (1 - m)
    list(id = pl$id, n = pl$n, m = m, lo = qbeta(0.1, a, b), hi = qbeta(0.9, a, b), mean = m * mu,
         ptsLo = .points_quantile(0.1, m, c, nodes, hi), ptsHi = .points_quantile(0.9, m, c, nodes, hi),
         lines = lapply(pl$lines, function(l) list(L = as.numeric(l$L), p = as.numeric(l$p),
                                                   pm = share_p_over(m, c, as.numeric(l$L), nodes))))
  })
  list(c = c, mu = mu, sd = sd, players = res, rest = 1 - sum(vapply(res, function(r) r$m, 0)), sse = sse)
}

# c, su kuriuo modelis geriausiai atkuria visas linijas (informatyvūs tik žaidėjai su ≥ 2 linijomis); NULL, jei tokių nėra.
# groups: sąrašas list(team = list(mu, sd), players = ...).
fit_concentration <- function(groups) {
  multi <- list()
  for (g in groups) for (p in g$players) if (length(p$lines) >= 2)
    multi[[length(multi) + 1]] <- list(nodes = team_nodes(as.numeric(g$team$mu), as.numeric(g$team$sd)), lines = p$lines)
  if (!length(multi)) return(NULL)
  obj <- function(c) sum(vapply(multi, function(x) fit_share(x$lines, c, x$nodes)$sse, 0))
  lo <- log(3); step <- (log(1500) - lo) / 15
  u <- lo + step * 0:15
  best <- u[which.min(vapply(u, function(x) obj(exp(x)), 0))]
  list(c = exp(.golden(function(x) obj(exp(x)), best - step, best + step, 20)), players = length(multi))
}

# Monte Carlo: rezultatas (dvimatis normalusis) ir žaidėjų taškai (Dirichlet dalys × komandos taškai) kartu.
# teams = list(home = list(c, players = list(list(id, m), ...)), away = ...); sąlygose v gali būti žaidėjo id.
player_condition_probability <- function(M, teams, conds, n = 40000, seed = 12345) {
  set.seed(seed)
  rho <- max(-0.95, min(0.95, M$rho))
  z1 <- rnorm(n); z2 <- rnorm(n)
  H <- pmax(0, round(M$mh + M$sh * z1))
  A <- pmax(0, round(M$ma + M$sa * (rho * z1 + sqrt(1 - rho^2) * z2)))
  P <- list()
  for (t in c("home", "away")) {
    T <- teams[[t]]; pl <- if (is.null(T)) list() else T$players
    if (!length(pl)) next
    cc <- as.numeric(T$c); ms <- vapply(pl, function(p) as.numeric(p$m), 0); tot <- sum(ms)
    scale <- if (tot > 0.98) 0.98 / tot else 1
    al <- c(cc * ms * scale, cc * max(0.02, 1 - tot * scale))
    g <- vapply(al, function(a) rgamma(n, a), numeric(n))
    g <- g / rowSums(g)
    pts <- if (t == "home") H else A
    for (j in seq_along(pl)) P[[pl[[j]]$id]] <- round(g[, j] * pts)
  }
  ok <- rep(TRUE, n)
  for (c in conds) {
    val <- if (!is.null(VARS[[c$v]])) VARS[[c$v]](H, A) else P[[c$v]]
    if (is.null(val) || is.null(OPS[[c$op]])) stop("Netinkama sąlyga: ", c$v, " ", c$op)
    ok <- ok & OPS[[c$op]](val, as.numeric(c$x))
  }
  p <- mean(ok)
  list(p = p, se = sqrt(p * (1 - p) / n))
}

# ---------- CSV ir komandinė eilutė ----------
read_rows <- function(path) {
  lines <- readLines(path, encoding = "UTF-8", warn = FALSE)
  out <- lapply(strsplit(lines, ","), function(cells) {
    nums <- suppressWarnings(as.numeric(sub(",", ".", trimws(cells))))
    nums <- nums[!is.na(nums)]
    if (length(nums) >= 3) nums[1:3] else NULL
  })
  out <- Filter(Negate(is.null), out)
  if (!length(out)) return(matrix(numeric(0), ncol = 3))
  do.call(rbind, out)
}

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  opt <- list(cond = list(), home_name = "Namai", away_name = "Svečiai", top = 10)
  i <- 1
  while (i <= length(args)) {
    key <- args[i]; val <- args[i + 1]
    switch(key,
      "--hcp" = , "--tot" = , "--home" = , "--away" = { opt[[sub("--", "", key)]] <- val },
      "--home-name" = { opt$home_name <- val },
      "--away-name" = { opt$away_name <- val },
      "--cond" = { opt$cond[[length(opt$cond) + 1]] <- parse_cond(val) },
      "--top" = { opt$top <- as.integer(val) },
      stop("Nežinomas parametras: ", key))
    i <- i + 2
  }
  F <- list()
  for (k in c("hcp", "tot", "home", "away"))
    if (!is.null(opt[[k]])) F[k] <- list(fit_market(k, read_rows(opt[[k]])))
  H <- opt$home_name; A <- opt$away_name
  if (!is.null(F$hcp))
    cat(sprintf("%s laimi: %.2f%%   skirtumas %+.1f (σ %.1f)\n", H, p_over(F$hcp, 0) * 100, F$hcp$mu, F$hcp$sigma))
  if (!is.null(F$tot)) cat(sprintf("Taškų suma: %.1f (σ %.1f)\n", F$tot$mu, F$tot$sigma))
  M <- score_model(F)
  if (is.null(M)) { cat("Rezultato modeliui reikia bent dviejų rinkų (pvz. handikapo ir sumos arba abiejų komandų taškų).\n"); return(invisible()) }
  cat(sprintf("Tikėtinas rezultatas: %s %.1f : %.1f %s   σ %.1f / %.1f   ρ = %.3f   (%s)\n",
              H, M$mh, M$ma, A, M$sh, M$sa, M$rho, M$src))
  alt <- c(if (is.finite(M$rho_tot)) sprintf("pagal sumą %.3f", M$rho_tot),
           if (is.finite(M$rho_hcp)) sprintf("pagal handikapą %.3f", M$rho_hcp))
  if (length(alt)) cat("ρ atskirai:", paste(alt, collapse = " · "), "\n")
  g <- grid_cells(M); g <- g[g$h != g$a, ]; g <- head(g[order(-g$p), ], opt$top)
  cat("\nTikėtiniausi rezultatai:\n")
  for (j in seq_len(nrow(g))) cat(sprintf("  %2d. %d : %d   %.2f%%\n", j, g$h[j], g$a[j], g$p[j] * 100))
  if (length(opt$cond)) {
    p <- condition_probability(M, opt$cond)
    txt <- paste(vapply(opt$cond, function(c) sprintf("%s %s %g", c$v, c$op, c$x), ""), collapse = " ir ")
    if (p > 0) cat(sprintf("\nP(%s) = %.2f%%   teisingas koeficientas %.2f\n", txt, p * 100, 1 / p))
    else cat(sprintf("\nP(%s) = 0\n", txt))
  }
}

if (sys.nframe() == 0) main()
