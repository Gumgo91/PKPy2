# rxode2 reference predictions for validate_pkpy2_extended_predictions.py.
# Each model is written as ODEs; dosing modifiers use rxode2's f(), alag(), dur(), rate().
suppressPackageStartupMessages({library(rxode2); library(jsonlite)})
rxode2::setRxThreads(1L)
root <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), ".."))
dir <- file.path(root, "output/pkpy2_extended_validation/predictions")
manifest <- fromJSON(file.path(dir, "manifest.json"), simplifyVector = FALSE)

models <- list(
  pk1iv = list(defaults = list(F = 1, ALAG = 0, D1 = 1, R1 = 1), code = "
    d/dt(central) = -CL/V*central
    f(central) = F
    alag(central) = ALAG
    dur(central) = D1
    rate(central) = R1
    cp = central/V
    out2 = cp"),
  pk1iv_tv = list(defaults = list(), code = "
    cl = TCL*(WT/70)^0.75
    d/dt(central) = -cl/V*central
    cp = central/V
    out2 = cp"),
  pk2iv = list(defaults = list(), code = "
    d/dt(central) = -(CL + Q)/V1*central + Q/V2*periph
    d/dt(periph) = Q/V1*central - Q/V2*periph
    cp = central/V1
    out2 = cp"),
  pk1oral = list(defaults = list(F = 1, ALAG = 0), code = "
    d/dt(depot) = -Ka*depot
    d/dt(central) = Ka*depot - CL/V*central
    f(depot) = F
    alag(depot) = ALAG
    cp = central/V
    out2 = cp"),
  pk2oral = list(defaults = list(F = 1, ALAG = 0), code = "
    d/dt(depot) = -Ka*depot
    d/dt(central) = Ka*depot - (CL + Q)/V1*central + Q/V2*periph
    d/dt(periph) = Q/V1*central - Q/V2*periph
    alag(depot) = ALAG
    cp = central/V1
    out2 = cp"),
  pk3iv = list(defaults = list(), code = "
    d/dt(central) = -(CL + Q2 + Q3)/V1*central + Q2/V2*p1 + Q3/V3*p2
    d/dt(p1) = Q2/V1*central - Q2/V2*p1
    d/dt(p2) = Q3/V1*central - Q3/V3*p2
    cp = central/V1
    out2 = cp"),
  transit3 = list(defaults = list(), code = "
    ktr = 4/MTT
    d/dt(t1) = -ktr*t1
    d/dt(t2) = ktr*t1 - ktr*t2
    d/dt(t3) = ktr*t2 - ktr*t3
    d/dt(depot) = ktr*t3 - Ka*depot
    d/dt(central) = Ka*depot - CL/V*central
    cp = central/V
    out2 = cp"),
  effect_sig = list(defaults = list(), code = "
    d/dt(depot) = -Ka*depot
    d/dt(central) = Ka*depot - CL/V*central
    d/dt(effect) = KE0*(central/V - effect)
    cp = central/V
    out2 = E0 + EMAX*effect^GAMMA/(EC50^GAMMA + effect^GAMMA)"),
  parent_met = list(defaults = list(), code = "
    d/dt(central) = -CL/V*central
    d/dt(metabolite) = FM*CL/V*central - CLM/VM*metabolite
    cp = central/V
    out2 = metabolite/VM"),
  mm1iv = list(defaults = list(), code = "
    d/dt(central) = -VMAX*(central/V)/(KM + central/V)
    cp = central/V
    out2 = cp"),
  mm1oral = list(defaults = list(), code = "
    d/dt(depot) = -Ka*depot
    d/dt(central) = Ka*depot - VMAX*(central/V)/(KM + central/V)
    cp = central/V
    out2 = cp"),
  tmdd_full = list(defaults = list(), code = "
    C = central/V
    bind = KON*C*target - KOFF*complex
    d/dt(central) = -CL/V*central - bind*V
    d/dt(target) = R0*KDEG - KDEG*target - bind
    d/dt(complex) = bind - KINT*complex
    target(0) = R0
    cp = C
    out2 = target + complex"),
  tmdd_qss = list(defaults = list(), code = "
    ctot = central/V
    b = ctot - rtot - KSS
    C = 0.5*(b + sqrt(b^2 + 4*KSS*ctot))
    d/dt(central) = -CL*C - KINT*(ctot - C)*V
    d/dt(rtot) = R0*KDEG - KDEG*rtot - (KINT - KDEG)*(ctot - C)
    rtot(0) = R0
    cp = C
    out2 = rtot")
)
idr <- function(kind) {
  eff <- if (kind <= 2) "IMAX*C/(IC50 + C)" else "EMAX*C/(EC50 + C)"
  eq <- switch(kind,
    "d/dt(R) = R0*KOUT*(1 - eff) - KOUT*R",
    "d/dt(R) = R0*KOUT - KOUT*(1 - eff)*R",
    "d/dt(R) = R0*KOUT*(1 + eff) - KOUT*R",
    "d/dt(R) = R0*KOUT - KOUT*(1 + eff)*R")
  list(defaults = list(), code = paste0("
    d/dt(depot) = -Ka*depot
    d/dt(central) = Ka*depot - CL/V*central
    C = central/V
    eff = ", eff, "
    ", eq, "
    R(0) = R0
    cp = C
    out2 = R"))
}
for (k in 1:4) models[[paste0("idr", k)]] <- idr(k)

for (sc in manifest) {
  spec <- models[[sc$r_model]]
  mod <- rxode2(spec$code)
  params <- modifyList(spec$defaults, sc$params)
  d <- read.csv(file.path(dir, paste0(sc$name, ".csv")), na.strings = ".")
  names(d) <- tolower(names(d))
  d$dv <- NULL
  keep <- intersect(names(d), c("id", "time", "evid", "amt", "cmt", "rate", "ii", "ss", "addl", "wt"))
  d <- d[, keep]
  res <- rxSolve(mod, unlist(params), d, atol = 1e-12, rtol = 1e-12, maxsteps = 5e6, ssAtol = 1e-12, ssRtol = 1e-12,
                 covsInterpolation = "locf", returnType = "data.frame")
  if (!"id" %in% names(res)) res$id <- d$id[1]
  obs <- d[d$evid == 0, ]
  res <- res[paste(res$id, res$time) %in% paste(obs$id, obs$time), ]   # drop EVID 2 rows
  out <- res[, c("id", "time", "cp", "out2")]
  write.csv(out, file.path(dir, paste0(sc$name, "_rxode2.csv")), row.names = FALSE)
  cat(sc$name, nrow(out), "\n")
}
