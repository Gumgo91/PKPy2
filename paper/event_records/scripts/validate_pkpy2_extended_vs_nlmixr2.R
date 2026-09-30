# nlmixr2 FOCEi fits of the simulated feature datasets (see validate_pkpy2_extended_vs_nlmixr2.py).
suppressPackageStartupMessages({library(nlmixr2); library(jsonlite)})
rxode2::setRxThreads(1L)
root <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), ".."))
dir <- file.path(root, "output/pkpy2_extended_validation/vs_nlmixr2")
models <- list(
  infusion_block_covariates = function() {
    ini({ tcl <- log(4); tv1 <- log(25); tq <- log(6); tv2 <- log(50); b.wt <- 0.5; b.sex <- -0.01
          eta.cl + eta.v1 ~ c(0.1, 0.01, 0.1); prop.sd <- 0.2; add.sd <- 0.1 })
    model({ cl <- exp(tcl + b.wt*log(WT/70) + eta.cl); v1 <- exp(tv1 + b.sex*SEX + eta.v1)
            q <- exp(tq); v2 <- exp(tv2); cp <- linCmt(); cp ~ add(add.sd) + prop(prop.sd) })
  },
  oral_blq_m3 = function() {
    ini({ tcl <- log(2.5); tv <- log(30); tka <- log(1); eta.cl ~ 0.1; eta.v ~ 0.1; eta.ka ~ 0.1; prop.sd <- 0.2 })
    model({ cl <- exp(tcl + eta.cl); v <- exp(tv + eta.v); ka <- exp(tka + eta.ka); cp <- linCmt(); cp ~ prop(prop.sd) })
  },
  lognormal_residual = function() {
    ini({ tcl <- log(3); tv <- log(30); eta.cl ~ 0.1; eta.v ~ 0.1; lnorm.sd <- 0.3 })
    model({ cl <- exp(tcl + eta.cl); v <- exp(tv + eta.v); cp <- linCmt(); cp ~ lnorm(lnorm.sd) })
  },
  michaelis_menten = function() {
    ini({ tv <- log(25); tvmax <- log(30); tkm <- log(3); eta.v ~ 0.1; eta.vmax ~ 0.1; prop.sd <- 0.2; add.sd <- 0.2 })
    model({ v <- exp(tv + eta.v); vmax <- exp(tvmax + eta.vmax); km <- exp(tkm)
            d/dt(central) <- -vmax*(central/v)/(km + central/v); cp <- central/v; cp ~ add(add.sd) + prop(prop.sd) })
  })
args <- commandArgs(trailingOnly = TRUE)
for (name in if (length(args)) args else names(models)) {
  d <- read.csv(file.path(dir, paste0(name, ".csv")), na.strings = ".")
  t0 <- proc.time()[["elapsed"]]
  fit <- nlmixr2(models[[name]], d, est = "focei", control = foceiControl(print = 0L, addProp = "combined2"))
  secs <- proc.time()[["elapsed"]] - t0
  om <- fit$omega
  cov <- list()
  if (nrow(om) > 1) for (a in 2:nrow(om)) for (b in 1:(a - 1)) cov[[paste0(rownames(om)[b], ",", rownames(om)[a])]] <- om[a, b]
  out <- list(scenario = name, objective = fit$objf, seconds = secs, theta = as.list(fit$theta),
              omega = setNames(as.list(diag(om)), rownames(om)), omega_cov = cov)
  write(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE), file.path(dir, paste0(name, "_nlmixr2.json")))
  cat(name, fit$objf, secs, "\n")
}
