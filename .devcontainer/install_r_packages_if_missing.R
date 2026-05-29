#!/usr/bin/env Rscript

cran_repo <- Sys.getenv("R_CRAN_REPO", "https://cloud.r-project.org")

cran_packages <- c(
  "arrow",
  "data.table",
  "ggplot2",
  "ggh4x",
  "gridExtra",
  "patchwork",
  "stringr",
  "future",
  "future.apply",
  "mvtnorm",
  "R.matlab",
  "Rmixmod"
)

missing_cran_packages <- cran_packages[
  !vapply(cran_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_cran_packages) > 0) {
  message(
    "Missing CRAN R packages: ",
    paste(missing_cran_packages, collapse = ", ")
  )
  install.packages(
    missing_cran_packages,
    repos = cran_repo,
    Ncpus = 4
  )
} else {
  message("No missing CRAN R packages.")
}

if (!requireNamespace("circhelp", quietly = TRUE)) {
  message("Missing GitHub R package: circhelp")
  if (!requireNamespace("remotes", quietly = TRUE)) {
    message("Missing CRAN R package required for GitHub install: remotes")
    install.packages("remotes", repos = cran_repo)
  }
  remotes::install_github("achetverikov/circhelp", upgrade = "never")
} else {
  message("No missing GitHub R packages.")
}
