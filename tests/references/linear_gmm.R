# External reference: free weights and component-specific diagonal covariance.
suppressPackageStartupMessages(library(Rmixmod))
args <- commandArgs(trailingOnly = TRUE)
options(digits = 17, warn = 2)

fit_from <- function(x, theta, iterations) {
  parameter <- new("GaussianParameter", proportions = theta[, 1],
                   mean = theta[, 2:3],
                   variance = lapply(1:2, function(k) diag(theta[k, 4:5]^2)))
  strategy <- mixmodStrategy(algo = "EM", initMethod = "parameter",
                            parameter = parameter, nbTry = 1,
                            nbIterationInAlgo = iterations, epsilonInAlgo = 1e-14)
  fit <- mixmodCluster(as.data.frame(x), nbCluster = 2,
                      models = mixmodGaussianModel(listModels = "Gaussian_pk_Lk_Bk"),
                      strategy = strategy)@bestResult
  p <- fit@parameters
  cbind(p@proportions, p@mean, t(vapply(p@variance, function(v) sqrt(diag(v)), numeric(2))),
        fit@likelihood)
}

for (name in args[-1]) {
  base <- file.path(args[1], name)
  x <- as.matrix(read.csv(paste0(base, "_data.csv"), header = FALSE))
  theta <- as.matrix(read.csv(paste0(base, "_initial.csv"), header = FALSE))
  one <- fit_from(x, theta, 1)
  converged <- fit_from(x, theta, 5000)
  extra <- fit_from(x, converged[, 1:5], 1)
  for (stage in c("one", "converged", "extra"))
    write.table(get(stage), paste0(base, "_r_", stage, ".csv"), sep = ",",
                row.names = FALSE, col.names = FALSE)
}
