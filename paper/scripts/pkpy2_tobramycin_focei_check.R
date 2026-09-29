# FOCEi (NONMEM-type) objective of nlmixr2 at the PKPy2 tobramycin solutions obtained with the
# expert's parameter bounds, and an nlmixr2 FOCEi fit with the same bounds from the documented
# initial values. Writes output/pkpy2_tobramycin_v3/focei_check.json.
suppressPackageStartupMessages(library(nlmixr2))
rxode2::setRxThreads(1L)
root <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), ".."))
d <- read.csv(file.path(root, "output/pkpy2_software_comparison/data/tobramycin.csv"), na.strings = ".")
d$AMT[is.na(d$AMT)] <- 0
tob <- file.path(root, "output/pkpy2_tobramycin_v3")

mk <- function(cl, v1, q, v2, bc, bw, om, sp) {
  eval(bquote(function() {
    ini({
      tcl <- c(-Inf, .(log(cl)), log(20)); tv1 <- c(-Inf, .(log(v1)), log(100)); tq <- c(-Inf, .(log(q)), log(50))
      tv2 <- c(-Inf, .(log(v2)), log(100)); b.clcr <- c(0, .(bc), 2); b.wt <- c(0, .(bw), 2)
      eta.cl ~ .(om); prop.sd <- .(sp); add.sd <- fix(0.0013601470508735443)
    })
    model({
      cl <- exp(tcl + b.clcr * LCLCR + eta.cl); vc <- exp(tv1 + b.wt * LWT); q <- exp(tq); vp <- exp(tv2)
      cp <- linCmt(); cp ~ add(add.sd) + prop(prop.sd)
    })
  }))
}
ctl <- function(outer) foceiControl(print = 0L, maxOuterIterations = outer, calcTables = FALSE, covMethod = "", addProp = "combined2")

at_point <- function(name) {
  f <- jsonlite::fromJSON(file.path(tob, paste0(name, ".json")))
  b <- pmin(pmax(f$coefficients, 1e-6), 2 - 1e-6)
  m <- mk(f$theta$CL, f$theta$V1, f$theta$Q, min(f$theta$V2, 100 - 1e-6), b[1], b[2], f$omega$CL, f$sigma$sigma_prop)
  e <- suppressMessages(nlmixr2(m, d, est = "focei", control = ctl(0L)))
  list(pkpy2_fit = name, pkpy2_exact_ofv = f$ofv, focei_objective = e$objf)
}
points <- lapply(c("bounded_start_1", "bounded_start_2"), at_point)
fit <- suppressMessages(nlmixr2(mk(3.4, 20.3, 5, 20, 1, 1, 0.1, 0.2), d, est = "focei", control = ctl(5000L)))
th <- fit$theta
out <- list(points = points,
            focei_fit_from_documented_start = list(objective = fit$objf, CL = exp(th[["tcl"]]), V1 = exp(th[["tv1"]]), Q = exp(th[["tq"]]),
                                                   V2 = exp(th[["tv2"]]), b_clcr = th[["b.clcr"]], b_wt = th[["b.wt"]]),
            published_expert_nonmem_ofv = 173.11,
            nlmixr2 = as.character(packageVersion("nlmixr2")))
write(jsonlite::toJSON(out, auto_unbox = TRUE, digits = NA, pretty = TRUE), file.path(tob, "focei_check.json"))
print(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 6))
