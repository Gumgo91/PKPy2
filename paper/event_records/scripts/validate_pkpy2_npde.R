# The npde package (Comets et al.) applied to PKPy2's observed data and simulated replicates.
suppressPackageStartupMessages(library(npde))
root <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), ".."))
dir <- file.path(root, "output/pkpy2_extended_validation/diagnostics")
obs <- read.csv(file.path(dir, "npde_obs.csv"))
sim <- read.csv(file.path(dir, "npde_sim.csv"))
res <- autonpde(obs, sim, iid = 1, ix = 2, iy = 3, boolsave = FALSE, verbose = FALSE, decorr.method = "cholesky",
                cens.method = "omit", ties = TRUE)
out <- data.frame(ID = obs$ID, TIME = obs$TIME, npde = res@results@res$npde, pd = res@results@res$pd)
write.csv(out, file.path(dir, "npde_package.csv"), row.names = FALSE)
cat("npde rows", nrow(out), "\n")
