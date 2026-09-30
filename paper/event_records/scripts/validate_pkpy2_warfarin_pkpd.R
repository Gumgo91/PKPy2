# nlmixr2 fits of the warfarin PK/PD turnover model (nlmixr2 example model, nlmixr2data::warfarin).
# Writes output/pkpy2_extended_validation/warfarin_pkpd/nlmixr2_<method>.json
suppressPackageStartupMessages({library(nlmixr2); library(jsonlite)})
rxode2::setRxThreads(1L)
args <- commandArgs(trailingOnly = TRUE)
method <- if (length(args)) args[1] else "focei"
root <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), ".."))
dir <- file.path(root, "output/pkpy2_extended_validation/warfarin_pkpd")
d <- nlmixr2data::warfarin

pk.turnover.emax <- function() {
  ini({
    tktr <- log(1)
    tka <- log(1)
    tcl <- log(0.1)
    tv <- log(10)
    eta.ktr ~ 1
    eta.ka ~ 1
    eta.cl ~ 2
    eta.v ~ 1
    prop.err <- 0.1
    pkadd.err <- 0.1
    temax <- logit(0.8)
    tec50 <- log(0.5)
    tkout <- log(0.05)
    te0 <- log(100)
    eta.emax ~ .5
    eta.ec50 ~ .5
    eta.kout ~ .5
    eta.e0 ~ .5
    pdadd.err <- 10
  })
  model({
    ktr <- exp(tktr + eta.ktr)
    ka <- exp(tka + eta.ka)
    cl <- exp(tcl + eta.cl)
    v <- exp(tv + eta.v)
    emax <- expit(temax + eta.emax)
    ec50 <- exp(tec50 + eta.ec50)
    kout <- exp(tkout + eta.kout)
    e0 <- exp(te0 + eta.e0)
    DCP <- center/v
    PD <- 1 - emax*DCP/(ec50 + DCP)
    effect(0) <- e0
    kin <- e0*kout
    d/dt(depot) <- -ktr*depot
    d/dt(gut) <- ktr*depot - ka*gut
    d/dt(center) <- ka*gut - cl/v*center
    d/dt(effect) <- kin*PD - kout*effect
    cp <- center/v
    cp ~ prop(prop.err) + add(pkadd.err)
    effect ~ add(pdadd.err) | pca
  })
}
ctl <- if (method == "focei") foceiControl(print = 0L, addProp = "combined2") else saemControl(print = 0L, addProp = "combined2")
t0 <- proc.time()[["elapsed"]]
fit <- nlmixr2(pk.turnover.emax, d, est = method, control = ctl)
secs <- proc.time()[["elapsed"]] - t0
th <- fit$theta
om <- fit$omega
out <- list(method = method, seconds = secs, objective = fit$objf,
            theta = as.list(th), omega = setNames(as.list(diag(om)), rownames(om)),
            nlmixr2 = as.character(packageVersion("nlmixr2")))
write(toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE), file.path(dir, paste0("nlmixr2_", method, ".json")))
print(toJSON(out, auto_unbox = TRUE, digits = 6))
