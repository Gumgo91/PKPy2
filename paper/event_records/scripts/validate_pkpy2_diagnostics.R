# nlmixr2 posthoc diagnostics at the PKPy2 theophylline estimates.
suppressPackageStartupMessages({library(nlmixr2); library(jsonlite)})
rxode2::setRxThreads(1L)
root <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), ".."))
dir <- file.path(root, "output/pkpy2_extended_validation/diagnostics")
est <- fromJSON(file.path(dir, "estimates.json"))
d <- read.csv(file.path(dir, "theophylline.csv"), na.strings = ".")
m <- eval(bquote(function() {
  ini({
    tcl <- .(log(est$theta$CL)); tv <- .(log(est$theta$V)); tka <- .(log(est$theta$Ka))
    eta.cl ~ .(est$omega$CL); eta.v ~ .(est$omega$V); eta.ka ~ .(est$omega$Ka)
    prop.sd <- .(est$sigma$sigma_prop)
  })
  model({
    cl <- exp(tcl + eta.cl) * WT/70
    v <- exp(tv + eta.v) * WT/70
    ka <- exp(tka + eta.ka)
    cp <- linCmt()
    cp ~ prop(prop.sd)
  })
}))
# FOCEi (interaction) conditional modes at the fixed estimates: zero outer iterations.
# (est = "posthoc" uses the residual variance at the population prediction, i.e. FOCE without interaction.)
fit <- nlmixr2(m, d, est = "focei", control = foceiControl(maxOuterIterations = 0L, interaction = TRUE, covMethod = "", print = 0L),
               table = tableControl(cwres = TRUE, npde = TRUE, nsim = 2000))
out <- as.data.frame(fit)
cat(names(out), "
"); keep <- intersect(c("ID", "TIME", "DV", "PRED", "IPRED", "RES", "IRES", "IWRES", "CWRES", "NPDE", "NPD", "EPRED", "ERES", "PDE", "eta.cl", "eta.v", "eta.ka"), names(out))
write.csv(out[, keep], file.path(dir, "nlmixr2_posthoc.csv"), row.names = FALSE)
cat("rows", nrow(out), "\n")
print(head(out[, keep]))
