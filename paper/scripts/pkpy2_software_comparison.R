# Head-to-head comparison of PKPy2 with established open-source NLME software.
#
# Estimators: nlmixr2 FOCEi, nlmixr2 SAEM, and saemix (SAEM).
# Inputs are the exact datasets analysed by PKPy2 (scripts/export_pkpy2_comparison_data.py).
# Models, starting values and fixed terms follow the PKPy2 protocols; nothing is tuned
# after inspecting results. One JSON record is written per dataset and estimator, and
# existing records are skipped so interrupted runs can be resumed.
#
# Usage: Rscript pkpy2_software_comparison.R simulation <shard> <n_shards>
#        Rscript pkpy2_software_comparison.R clinical

suppressPackageStartupMessages({
  library(nlmixr2)
  library(saemix)
  library(jsonlite)
})
rxode2::setRxThreads(1L)

args <- commandArgs(trailingOnly = TRUE)
task <- if (length(args)) args[[1]] else "simulation"
shard <- if (length(args) >= 2) as.integer(args[[2]]) else 0L
n_shards <- if (length(args) >= 3) as.integer(args[[3]]) else 1L

root <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), ".."))
base <- file.path(root, "output", "pkpy2_software_comparison")
data_dir <- file.path(base, "data")
SEED <- 260923L

read_nm <- function(path) {
  d <- read.csv(path, na.strings = ".", stringsAsFactors = FALSE)
  d$AMT[is.na(d$AMT)] <- 0
  if (all(is.na(d$WT))) d$WT <- NULL
  d
}

obs_frame <- function(d) {
  dose <- d[d$EVID == 1, c("ID", "AMT")]
  o <- d[d$EVID == 0, ]
  o$DOSE <- dose$AMT[match(o$ID, dose$ID)]
  o
}

safe_se <- function(cov, name) {
  if (is.null(cov) || !(name %in% rownames(cov))) return(NA_real_)
  v <- cov[name, name]
  if (!is.finite(v) || v < 0) NA_real_ else sqrt(v)
}

# ---------------------------------------------------------------- nlmixr2 models
m_iv <- function() {
  ini({
    tcl <- log(3.2)
    tv <- log(32)
    eta.cl ~ 0.08
    eta.v ~ 0.08
    prop.sd <- 0.2
  })
  model({
    cl <- exp(tcl + eta.cl)
    v <- exp(tv + eta.v)
    cp <- linCmt()
    cp ~ prop(prop.sd)
  })
}

m_theo <- function() {
  ini({
    tcl <- log(3)
    tv <- log(30)
    tka <- log(1)
    eta.cl ~ 0.1
    eta.v ~ 0.03
    eta.ka ~ 0.3
    prop.sd <- 0.15
  })
  model({
    cl <- exp(tcl + eta.cl) * (WT / 70)
    v <- exp(tv + eta.v) * (WT / 70)
    ka <- exp(tka + eta.ka)
    cp <- linCmt()
    cp ~ prop(prop.sd)
  })
}

m_warf <- function() {
  ini({
    tcl <- log(0.2)
    tv <- log(10)
    tka <- log(0.8)
    tlag <- log(0.5)
    eta.cl ~ 0.1
    eta.ka ~ fix(0.5)
    prop.sd <- 0.2
    add.sd <- fix(0.3420526275297414)
  })
  model({
    cl <- exp(tcl + eta.cl) * (WT / 70)^0.75
    v <- exp(tv) * (WT / 70)
    ka <- exp(tka + eta.ka)
    lag <- exp(tlag)
    d/dt(depot) <- -ka * depot
    d/dt(central) <- ka * depot - cl / v * central
    alag(depot) <- lag
    cp <- central / v
    cp ~ add(add.sd) + prop(prop.sd)
  })
}

# Tobramycin: two-compartment IV, power covariates written in mu-referenced form
# (LCLCR = log(CLCR/58), LWT = log(WT/62)), IIV on CL, additive variance fixed at 1.85e-6.
m_tobra <- function() {
  ini({
    tcl <- log(3.4)
    tv1 <- log(20.3)
    tq <- log(5)
    tv2 <- log(20)
    b.clcr <- 1
    b.wt <- 1
    eta.cl ~ 0.1
    prop.sd <- 0.2
    add.sd <- fix(0.0013601470508735443)
  })
  model({
    cl <- exp(tcl + b.clcr * LCLCR + eta.cl)
    vc <- exp(tv1 + b.wt * LWT)
    q <- exp(tq)
    vp <- exp(tv2)
    cp <- linCmt()
    cp ~ add(add.sd) + prop(prop.sd)
  })
}

# Reduced tobramycin model: WT exponent on V1 fixed at 1.
m_tobra_reduced <- function() {
  ini({
    tcl <- log(3.4)
    tv1 <- log(20.3)
    tq <- log(5)
    tv2 <- log(20)
    b.clcr <- 1
    eta.cl ~ 0.1
    prop.sd <- 0.2
    add.sd <- fix(0.0013601470508735443)
  })
  model({
    cl <- exp(tcl + b.clcr * LCLCR + eta.cl)
    vc <- exp(tv1 + LWT)
    q <- exp(tq)
    vp <- exp(tv2)
    cp <- linCmt()
    cp ~ add(add.sd) + prop(prop.sd)
  })
}

# Tobramycin with the expert's judgment as constraints: WT exponent on V1 fixed at 1,
# V1 <= 10 L, V2 <= 30 L, CL <= 20 L/h, Q <= 50 L/h, CLCR exponent in (0, 2).
# Start: documented initial values with V1 = 8 L (within the constraint).
m_tobra_expert <- function() {
  ini({
    tcl <- c(-Inf, log(3.4), log(20))
    tv1 <- c(-Inf, log(8), log(10))
    tq <- c(-Inf, log(5), log(50))
    tv2 <- c(-Inf, log(20), log(30))
    b.clcr <- c(0, 1, 2)
    eta.cl ~ 0.1
    prop.sd <- 0.2
    add.sd <- fix(0.0013601470508735443)
  })
  model({
    cl <- exp(tcl + b.clcr * LCLCR + eta.cl)
    vc <- exp(tv1 + LWT)
    q <- exp(tq)
    vp <- exp(tv2)
    cp <- linCmt()
    cp ~ add(add.sd) + prop(prop.sd)
  })
}

# Map nlmixr2 coordinates to PKPy2 reporting names.
theta_map <- c(tcl = "CL", tv = "V", tka = "Ka", tlag = "ALAG", tv1 = "V1", tq = "Q", tv2 = "V2")
coef_names <- c("b.clcr", "b.wt")
omega_map <- c(eta.cl = "CL", eta.v = "V", eta.ka = "Ka")

fit_nlmixr2 <- function(model, data, est) {
  ctl <- if (est == "focei") {
    foceiControl(print = 0L, calcTables = FALSE, addProp = "combined2")
  } else {
    saemControl(print = 0L, calcTables = FALSE, seed = SEED, addProp = "combined2")
  }
  warnings_seen <- character()
  t0 <- proc.time()[["elapsed"]]
  fit <- tryCatch(
    withCallingHandlers(
      suppressMessages(nlmixr2(model, data, est = est, control = ctl)),
      warning = function(w) {
        warnings_seen <<- c(warnings_seen, conditionMessage(w))
        invokeRestart("muffleWarning")
      }),
    error = function(e) e)
  seconds <- proc.time()[["elapsed"]] - t0
  if (inherits(fit, "error")) {
    return(list(status = "error", message = conditionMessage(fit), seconds = seconds,
                warnings = unique(warnings_seen)))
  }
  th <- fit$theta
  cov <- tryCatch(fit$cov, error = function(e) NULL)
  om <- fit$omega
  theta <- list(); se_log <- list(); omega <- list(); sigma <- list(); se_sigma <- list()
  for (k in names(theta_map)) if (k %in% names(th)) {
    theta[[theta_map[[k]]]] <- exp(th[[k]])
    se_log[[theta_map[[k]]]] <- safe_se(cov, k)
  }
  for (k in names(omega_map)) if (k %in% rownames(om)) omega[[omega_map[[k]]]] <- om[k, k]
  coefficients <- list(); se_coefficients <- list()
  for (k in coef_names) if (k %in% names(th)) {
    coefficients[[k]] <- th[[k]]
    se_coefficients[[k]] <- safe_se(cov, k)
  }
  for (k in c("prop.sd", "add.sd")) if (k %in% names(th)) {
    sigma[[sub(".sd", "", k, fixed = TRUE)]] <- abs(th[[k]])
    se_sigma[[sub(".sd", "", k, fixed = TRUE)]] <- safe_se(cov, k)
  }
  list(status = "returned", seconds = seconds, objective = fit$objf,
       theta = theta, se_log_theta = se_log, omega = omega, sigma = sigma, se_sigma = se_sigma,
       coefficients = coefficients, se_coefficients = se_coefficients,
       covariance_method = tryCatch(as.character(fit$covMethod), error = function(e) NA_character_),
       covariance_available = !is.null(cov),
       message = tryCatch(paste(fit$message, collapse = " "), error = function(e) ""),
       warnings = unique(warnings_seen))
}

# ---------------------------------------------------------------- saemix models
saemix_iv <- function(psi, id, xidep) {
  dose <- xidep[, 1]; t <- xidep[, 2]
  cl <- psi[id, 1]; v <- psi[id, 2]
  dose / v * exp(-cl / v * t)
}

saemix_theo <- function(psi, id, xidep) {
  dose <- xidep[, 1]; t <- xidep[, 2]; wt <- xidep[, 3]
  cl <- psi[id, 1] * wt / 70; v <- psi[id, 2] * wt / 70; ka <- psi[id, 3]
  k <- cl / v
  dose * ka / (v * (ka - k)) * (exp(-k * t) - exp(-ka * t))
}

fit_saemix <- function(kind, d) {
  o <- obs_frame(d)
  if (kind == "iv") {
    sd <- saemixData(name.data = o, name.group = "ID", name.predictors = c("DOSE", "TIME"),
                     name.response = "DV", verbose = FALSE)
    sm <- saemixModel(model = saemix_iv, psi0 = matrix(c(3.2, 32), ncol = 2, dimnames = list(NULL, c("CL", "V"))),
                      transform.par = c(1, 1), covariance.model = diag(2), omega.init = diag(c(0.08, 0.08)),
                      error.model = "proportional", error.init = c(0, 0.2), verbose = FALSE)
    names_psi <- c("CL", "V")
  } else {
    sd <- saemixData(name.data = o, name.group = "ID", name.predictors = c("DOSE", "TIME", "WT"),
                     name.response = "DV", verbose = FALSE)
    sm <- saemixModel(model = saemix_theo, psi0 = matrix(c(3, 30, 1), ncol = 3, dimnames = list(NULL, c("CL", "V", "Ka"))),
                      transform.par = c(1, 1, 1), covariance.model = diag(3), omega.init = diag(c(0.1, 0.03, 0.3)),
                      error.model = "proportional", error.init = c(0, 0.15), verbose = FALSE)
    names_psi <- c("CL", "V", "Ka")
  }
  opt <- list(seed = SEED, displayProgress = FALSE, print = FALSE, save = FALSE, save.graphs = FALSE,
              warnings = FALSE, fim = TRUE, ll.is = FALSE, map = FALSE)
  warnings_seen <- character()
  t0 <- proc.time()[["elapsed"]]
  fit <- tryCatch(
    withCallingHandlers(
      suppressMessages(capture.output(res <- saemix(sm, sd, opt))),
      warning = function(w) {
        warnings_seen <<- c(warnings_seen, conditionMessage(w))
        invokeRestart("muffleWarning")
      }),
    error = function(e) e)
  seconds <- proc.time()[["elapsed"]] - t0
  if (inherits(fit, "error")) {
    return(list(status = "error", message = conditionMessage(fit), seconds = seconds,
                warnings = unique(warnings_seen)))
  }
  r <- res@results
  fe <- r@fixed.effects; sef <- r@se.fixed
  theta <- list(); se_log <- list(); omega <- list()
  for (j in seq_along(names_psi)) {
    theta[[names_psi[j]]] <- fe[j]
    se_log[[names_psi[j]]] <- if (length(sef) >= j && is.finite(sef[j])) sef[j] / fe[j] else NA_real_
    omega[[names_psi[j]]] <- r@omega[j, j]
  }
  rp <- r@respar; serp <- r@se.respar
  list(status = "returned", seconds = seconds, objective = NA_real_,
       theta = theta, se_log_theta = se_log, omega = omega,
       sigma = list(prop = rp[2]), se_sigma = list(prop = if (length(serp) >= 2) serp[2] else NA_real_),
       covariance_available = length(sef) > 0 && all(is.finite(sef)),
       message = "", warnings = unique(warnings_seen))
}

# saemix, tobramycin: two-compartment IV bolus superposition over each subject's dose history.
# saemix cannot fix the additive residual term, so a proportional error model is used
# (the fixed additive SD of the reference model is 0.0014 mg/L).
fit_saemix_tobra <- function(d, fix_wt = FALSE) {
  o <- d[d$EVID == 0, ]
  o$OBS <- seq_len(nrow(o))
  o$LWTP <- o$LWT
  doses <- d[d$EVID == 1, c("ID", "TIME", "AMT")]
  long <- merge(o[, c("ID", "TIME", "OBS")], doses, by = "ID", suffixes = c("", ".dose"))
  long <- long[long$TIME.dose <= long$TIME, ]
  L_obs <- long$OBS; L_dt <- long$TIME - long$TIME.dose; L_amt <- long$AMT
  model <- function(psi, id, xidep) {
    obs <- xidep[, 2]
    m <- match(L_obs, obs)
    keep <- !is.na(m)
    m <- m[keep]; dt <- L_dt[keep]; amt <- L_amt[keep]
    p <- psi[id[m], , drop = FALSE]
    cl <- p[, 1]; v1 <- p[, 2]; q <- p[, 3]; v2 <- p[, 4]
    if (fix_wt) v1 <- v1 * exp(xidep[m, 3])
    k10 <- cl / v1; k12 <- q / v1; k21 <- q / v2
    s <- k10 + k12 + k21
    root <- sqrt(s * s - 4 * k10 * k21)
    alpha <- (s + root) / 2; beta <- (s - root) / 2
    unit <- ((alpha - k21) * exp(-alpha * dt) + (k21 - beta) * exp(-beta * dt)) / ((alpha - beta) * v1)
    out <- numeric(length(obs))
    agg <- rowsum(amt * unit, m)
    out[as.integer(rownames(agg))] <- agg[, 1]
    out
  }
  sd <- saemixData(name.data = o, name.group = "ID", name.response = "DV",
                   name.predictors = if (fix_wt) c("TIME", "OBS", "LWTP") else c("TIME", "OBS"),
                   name.covariates = if (fix_wt) "LCLCR" else c("LCLCR", "LWT"), verbose = FALSE)
  psi0 <- matrix(c(3.4, 20.3, 5, 20, 1, if (fix_wt) 0 else 1, 0, 0), nrow = 2, byrow = TRUE,
                 dimnames = list(NULL, c("CL", "V1", "Q", "V2")))
  sm <- saemixModel(model = model, psi0 = psi0, transform.par = c(1, 1, 1, 1),
                    covariate.model = if (fix_wt) matrix(c(1, 0, 0, 0), nrow = 1) else
                      matrix(c(1, 0, 0, 0, 0, 1, 0, 0), nrow = 2, byrow = TRUE),
                    covariance.model = diag(c(1, 0, 0, 0)), omega.init = diag(c(0.1, 1, 1, 1)),
                    error.model = "proportional", error.init = c(0, 0.2), verbose = FALSE)
  opt <- list(seed = SEED, displayProgress = FALSE, print = FALSE, save = FALSE, save.graphs = FALSE,
              warnings = FALSE, fim = TRUE, ll.is = FALSE, map = FALSE)
  warnings_seen <- character()
  t0 <- proc.time()[["elapsed"]]
  fit <- tryCatch(
    withCallingHandlers(
      suppressMessages(capture.output(res <- saemix(sm, sd, opt))),
      warning = function(w) {
        warnings_seen <<- c(warnings_seen, conditionMessage(w))
        invokeRestart("muffleWarning")
      }),
    error = function(e) e)
  seconds <- proc.time()[["elapsed"]] - t0
  if (inherits(fit, "error")) {
    return(list(status = "error", message = conditionMessage(fit), seconds = seconds,
                warnings = unique(warnings_seen)))
  }
  r <- res@results
  ci <- r@conf.int
  get <- function(name, col) { v <- ci[ci$name == name, col]; if (length(v)) v[[1]] else NA_real_ }
  theta <- list(); se_log <- list()
  for (n in c("CL", "V1", "Q", "V2")) {
    theta[[n]] <- get(n, "estimate")
    se_log[[n]] <- get(n, "se") / get(n, "estimate")
  }
  b_clcr <- ci$name[grepl("LCLCR", ci$name)][1]
  b_wt <- ci$name[grepl("LWT", ci$name)][1]
  coefficients <- list(b.clcr = get(b_clcr, "estimate"), b.wt = if (fix_wt) 1 else get(b_wt, "estimate"))
  se_coefficients <- list(b.clcr = get(b_clcr, "se"), b.wt = if (fix_wt) NA_real_ else get(b_wt, "se"))
  list(status = "returned", seconds = seconds, objective = NA_real_,
       theta = theta, se_log_theta = se_log, omega = list(CL = r@omega[1, 1]),
       sigma = list(prop = r@respar[2]), se_sigma = list(prop = if (length(r@se.respar) >= 2) r@se.respar[2] else NA_real_),
       coefficients = coefficients, se_coefficients = se_coefficients, confidence_table_names = ci$name,
       covariance_available = all(is.finite(ci$se[!is.na(ci$se)])), message = "", warnings = unique(warnings_seen))
}

write_record <- function(path, record) {
  write(toJSON(record, auto_unbox = TRUE, digits = NA, null = "null", na = "null", pretty = TRUE), path)
}

environment_record <- function() {
  list(R = R.version.string,
       nlmixr2 = as.character(packageVersion("nlmixr2")),
       nlmixr2est = as.character(packageVersion("nlmixr2est")),
       rxode2 = as.character(packageVersion("rxode2")),
       saemix = as.character(packageVersion("saemix")),
       platform = R.version$platform, seed = SEED, rx_threads = 1L)
}

run_one <- function(out_dir, dataset, estimator, fun, extra = list()) {
  path <- file.path(out_dir, paste0(dataset, "__", estimator, ".json"))
  if (file.exists(path)) return(invisible(NULL))
  res <- fun()
  write_record(path, c(list(dataset = dataset, estimator = estimator), extra, res,
                       list(environment = environment_record())))
  cat(format(Sys.time()), dataset, estimator, res$status, sprintf("%.1fs", res$seconds), "\n")
}

if (task == "simulation") {
  out_dir <- file.path(base, "results", "simulation")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  manifest <- fromJSON(file.path(data_dir, "manifest.json"))
  sims <- manifest[grepl("^simulation/", manifest$file), ]
  idx <- seq_len(nrow(sims))
  idx <- idx[(idx - 1L) %% n_shards == shard]
  for (i in idx) {
    row <- sims[i, ]
    d <- read_nm(file.path(data_dir, row$file))
    extra <- list(sampling = row$sampling, replicate = row$replicate, seed = row$seed)
    run_one(out_dir, row$dataset, "nlmixr2_focei", function() fit_nlmixr2(m_iv, d, "focei"), extra)
    run_one(out_dir, row$dataset, "nlmixr2_saem", function() fit_nlmixr2(m_iv, d, "saem"), extra)
    run_one(out_dir, row$dataset, "saemix", function() fit_saemix("iv", d), extra)
  }
} else if (task == "clinical") {
  out_dir <- file.path(base, "results", "clinical")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  theo <- read_nm(file.path(data_dir, "theophylline.csv"))
  warf <- read_nm(file.path(data_dir, "warfarin.csv"))
  run_one(out_dir, "theophylline", "nlmixr2_focei", function() fit_nlmixr2(m_theo, theo, "focei"))
  run_one(out_dir, "theophylline", "nlmixr2_saem", function() fit_nlmixr2(m_theo, theo, "saem"))
  run_one(out_dir, "theophylline", "saemix", function() fit_saemix("theo", theo))
  run_one(out_dir, "warfarin", "nlmixr2_focei", function() fit_nlmixr2(m_warf, warf, "focei"))
  run_one(out_dir, "warfarin", "nlmixr2_saem", function() fit_nlmixr2(m_warf, warf, "saem"))
} else if (task == "tobramycin_expert") {
  out_dir <- file.path(base, "results", "clinical")
  tobra <- read_nm(file.path(data_dir, "tobramycin.csv"))
  run_one(out_dir, "tobramycin_expert", "nlmixr2_focei", function() fit_nlmixr2(m_tobra_expert, tobra, "focei"))
  run_one(out_dir, "tobramycin_expert", "nlmixr2_saem", function() fit_nlmixr2(m_tobra_expert, tobra, "saem"))
} else if (task == "tobramycin_reduced") {
  out_dir <- file.path(base, "results", "clinical")
  tobra <- read_nm(file.path(data_dir, "tobramycin.csv"))
  run_one(out_dir, "tobramycin_reduced", "nlmixr2_focei", function() fit_nlmixr2(m_tobra_reduced, tobra, "focei"))
  run_one(out_dir, "tobramycin_reduced", "nlmixr2_saem", function() fit_nlmixr2(m_tobra_reduced, tobra, "saem"))
  run_one(out_dir, "tobramycin_reduced", "saemix", function() fit_saemix_tobra(tobra, fix_wt = TRUE))
} else if (task == "tobramycin") {
  out_dir <- file.path(base, "results", "clinical")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  tobra <- read_nm(file.path(data_dir, "tobramycin.csv"))
  run_one(out_dir, "tobramycin", "nlmixr2_focei", function() fit_nlmixr2(m_tobra, tobra, "focei"))
  run_one(out_dir, "tobramycin", "nlmixr2_saem", function() fit_nlmixr2(m_tobra, tobra, "saem"))
  run_one(out_dir, "tobramycin", "saemix", function() fit_saemix_tobra(tobra))
} else {
  stop("unknown task: ", task)
}
