#' Surface Simulator R Interface
#'
#' Wraps surface_simulator.py: writes a parameter Arrow file, calls Python,
#' reads back the long-format results.

library(arrow)
library(stringr)
library(data.table)

# Locate the Python script and repo root once, at source time.
# .sim_script_dir  — directory containing this R file and surface_simulator.py
# .sim_repo_root   — parent directory (repo root); Python is run from here so
#                    that relative paths like "results/" resolve correctly.
.sim_script_dir <- tryCatch(
  normalizePath(dirname(sys.frame(1)$ofile)),
  error = function(e) getwd()
)
.sim_py_script <- file.path(.sim_script_dir, "surface_simulator.py")
.sim_repo_root <- normalizePath(dirname(.sim_script_dir))
.sim_workspace_root <- normalizePath(file.path(.sim_repo_root, ".."))

#' Simulate surfaces for given parameter combinations
#'
#' @param parameters Data frame with columns: sd_feat1, sd_feat2, sd_spat, condition_id
#'                  (sd_motor optional - will be set to 0 if skip_motor_noise=TRUE)
#' @param n_samples Number of samples used for training (determines which model to use)
#' @param skip_motor_noise Whether to skip motor noise computation (default: FALSE)
#' @param use_nn_surfaces Whether to use NN predictions (TRUE) or averaged surfaces from files (FALSE) (default: TRUE)
#'                       Note: NN surfaces only provide mu1 bias curves. For mu2 bias curves, use averaged surfaces (FALSE)
#' @param averaged_surfaces_dir Path to the averaged surfaces directory (required when use_nn_surfaces=FALSE)
#' @param cleanup Whether to clean up temporary files (default: TRUE)
#' @param work_dir Directory for temporary input/output Arrow files (default: getwd()).
#'                 Any path conversion needed to reach the Python process (e.g. WSL
#'                 mnt paths) should be handled by the caller before passing work_dir.
#' @return Data frame containing simulation results with columns: sd_feat1, sd_feat2, sd_spat, sd_motor,
#'         feat_diff, mu1_density_asymmetry, mu2_density_asymmetry (if available), mu1_expectation,
#'         mu2_expectation (if available), sd_curve
simulate_surfaces <- function(parameters,
                              n_samples,
                              skip_motor_noise      = FALSE,
                              use_nn_surfaces       = TRUE,
                              averaged_surfaces_dir = NULL,
                              checkpoint_path       = NULL,
                              cleanup               = TRUE,
                              work_dir              = getwd()) {

  session_id    <- format(Sys.time(), "%Y%m%d_%H%M%S")
  input_file_r  <- file.path(work_dir, paste0("params_",  session_id, ".arrow"))
  output_file_r <- file.path(work_dir, paste0("results_", session_id, ".arrow"))

  cat("work_dir:", work_dir, "\n")

  # Validate inputs
  if (!use_nn_surfaces && is.null(averaged_surfaces_dir)) {
    stop("averaged_surfaces_dir is required when use_nn_surfaces = FALSE")
  }
  if (!is.null(averaged_surfaces_dir) && !grepl("^/", averaged_surfaces_dir)) {
    averaged_surfaces_dir <- normalizePath(
      file.path(.sim_workspace_root, averaged_surfaces_dir),
      mustWork = FALSE
    )
  }
  if (!is.null(checkpoint_path) && !grepl("^/", checkpoint_path)) {
    checkpoint_path <- normalizePath(
      file.path(.sim_workspace_root, checkpoint_path),
      mustWork = FALSE
    )
  }

  if (skip_motor_noise) {
    required_cols <- c("sd_feat1", "sd_feat2", "sd_spat")
  } else {
    required_cols <- c("sd_feat1", "sd_feat2", "sd_spat", "sd_motor")
  }

  missing_cols <- setdiff(required_cols, names(parameters))
  if (length(missing_cols) > 0) {
    stop("Missing required columns in parameters: ", paste(missing_cols, collapse = ", "))
  }

  cat("Creating parameter file:", input_file_r, "\n")
  write_parquet(parameters, input_file_r)

  # Build Python command — absolute paths for I/O, cd to repo root so that
  # "results/" and other relative paths in surface_simulator.py resolve correctly.
  pythonpath_parts <- c(
    .sim_repo_root,
    file.path(.sim_repo_root, "neural_network_optimization"),
    Sys.getenv("PYTHONPATH", unset = "")
  )
  pythonpath <- paste(pythonpath_parts[nzchar(pythonpath_parts)], collapse = ":")
  cmd_args <- c(
    shQuote(input_file_r),
    n_samples,
    shQuote(output_file_r),
    if (skip_motor_noise) "--skip-motor-noise",
    "--surface-source", if (use_nn_surfaces) "nn" else "raw",
    if (!use_nn_surfaces) c("--averaged-surfaces-dir", shQuote(averaged_surfaces_dir)),
    if (!is.null(checkpoint_path)) c("--checkpoint-path", shQuote(checkpoint_path))
  )

  full_cmd <- paste(
    "cd", shQuote(.sim_repo_root), "&&",
    sprintf("PYTHONPATH=%s", shQuote(pythonpath)),
    "python", shQuote(.sim_py_script), paste(cmd_args, collapse = " ")
  )

  cat("Running:", full_cmd, "\n")
  result <- system(full_cmd, intern = TRUE)
  
  # Check if command executed successfully
  exit_code <- attr(result, "status")
  if (!is.null(exit_code) && exit_code != 0) {
    stop("Python simulation failed with exit code: ", exit_code, "\n", 
         "Output: ", paste(result, collapse = "\n"))
  }
  
  cat("Python output:\n")
  cat(paste(result, collapse = "\n"), "\n")
  
  # Wait a bit to ensure file is written
  Sys.sleep(2)
  
  # Check if output file exists
  if (!file.exists(output_file_r)) {
    stop("Output file not found: ", output_file_r)
  }
  
  cat("Loading results from:", output_file_r, "\n")
  
  # Load Arrow results
  tryCatch({
    sim_results <- read_parquet(output_file_r)
    
    # Extract metadata from first row (where it's not NULL)
    metadata <- list(
      feat_diff_grid = unlist(sim_results$feat_diff_grid[1]),
      mu1_bias_grid = unlist(sim_results$mu1_bias_grid[1]),
      mu2_bias_grid = if(!is.null(sim_results$mu2_bias_grid[1])) unlist(sim_results$mu2_bias_grid[1]) else NULL,
      feat_diff_range = unlist(sim_results$feat_diff_range[1]),
      mu1_bias_range = unlist(sim_results$mu1_bias_range[1]),
      mu2_bias_range = if(!is.null(sim_results$mu2_bias_range[1])) unlist(sim_results$mu2_bias_range[1]) else NULL,
      feat_diff_step = sim_results$feat_diff_step[1],
      mu1_bias_step = sim_results$mu1_bias_step[1],
      mu2_bias_step = if(!is.null(sim_results$mu2_bias_step[1])) sim_results$mu2_bias_step[1] else NULL,
      n_samples = sim_results$n_samples[1],
      skip_motor_noise = sim_results$skip_motor_noise[1],
      has_mu2_data = if(!is.null(sim_results$has_mu2_data[1])) sim_results$has_mu2_data[1] else FALSE
    )
    
    # Unpack to long format
    cat("Unpacking to long format...\n")
    
    # Get parameter columns
    param_cols <- c("sd_feat1", "sd_feat2", "sd_spat", "sd_motor")
    
    # Create long format data
    long_results <- data.frame()
    
    for (i in 1:nrow(sim_results)) {
      # Get parameters for this row
      params <- sim_results[i, param_cols]
      
      # Get density curves, expectation curves, and SD curve
      # Handle both old and new column names for mu1 density
      if("mu1_density_curve" %in% names(sim_results)) {
        mu1_density_curve <- unlist(sim_results$mu1_density_curve[i])
      } else if("density_curve" %in% names(sim_results)) {
        mu1_density_curve <- unlist(sim_results$density_curve[i])
      } else {
        mu1_density_curve <- rep(NA, length(metadata$feat_diff_grid))
      }
      
      # Handle mu2 density curve
      mu2_density_curve <- if("mu2_density_curve" %in% names(sim_results) && !is.null(sim_results$mu2_density_curve[i])) {
        unlist(sim_results$mu2_density_curve[i])
      } else {
        rep(NA, length(metadata$feat_diff_grid))
      }
      
      # Handle both old and new column names for mu1 expectation
      if("mu1_expectation_curve" %in% names(sim_results)) {
        mu1_expectation_curve <- unlist(sim_results$mu1_expectation_curve[i])
      } else if("expectation_curve" %in% names(sim_results)) {
        mu1_expectation_curve <- unlist(sim_results$expectation_curve[i])
      } else {
        mu1_expectation_curve <- rep(NA, length(metadata$feat_diff_grid))
      }
      
      # Handle mu2 expectation curve
      mu2_expectation_curve <- if("mu2_expectation_curve" %in% names(sim_results) && !is.null(sim_results$mu2_expectation_curve[i])) {
        unlist(sim_results$mu2_expectation_curve[i])
      } else {
        rep(NA, length(metadata$feat_diff_grid))
      }
      sd_curve <- unlist(sim_results$sd_curve[i])
      
      # Create data frame for this parameter combination
      # All curves now use the same feat_diff_grid
      row_data <- data.frame(
        sd_feat1 = params$sd_feat1,
        sd_feat2 = params$sd_feat2,
        sd_spat = params$sd_spat,
        sd_motor = params$sd_motor,
        feat_diff = metadata$feat_diff_grid,
        mu1_density_asymmetry = mu1_density_curve,
        mu2_density_asymmetry = mu2_density_curve,
        mu1_expectation = mu1_expectation_curve,
        mu2_expectation = mu2_expectation_curve,
        sd_curve = sd_curve
      )
      
      long_results <- rbind(long_results, row_data)
    }
    
    # Add metadata as attributes
    attr(long_results, "metadata") <- metadata
    
    cat("Successfully unpacked", nrow(long_results), "data points from", nrow(sim_results), "parameter combinations\n")
    
    # Cleanup temporary files if requested
    if (cleanup) {
      unlink(output_file_r)
      unlink(input_file_r)
      cat("Cleaned up temporary files\n")
    }
    
    return(long_results)
    
  }, error = function(e) {
    stop("Failed to load results: ", e$message)
  })
}

simulate_unequal_noise2 <- function(sd_feat_range         = seq(10, 60, 10),
                                    sd_spat               = 42,
                                    n_samples             = 20,
                                    use_nn_surfaces       = TRUE,
                                    averaged_surfaces_dir = NULL,
                                    checkpoint_path       = NULL,
                                    work_dir              = getwd()) {
  par_grid <- expand.grid(sd_feat1 = sd_feat_range, sd_feat2 = sd_feat_range, sd_spat = sd_spat)
  res <- simulate_surfaces(par_grid,
                           n_samples             = n_samples,
                           skip_motor_noise      = TRUE,
                           use_nn_surfaces       = use_nn_surfaces,
                           averaged_surfaces_dir = averaged_surfaces_dir,
                           checkpoint_path       = checkpoint_path,
                           work_dir              = work_dir)
  setDT(res)
  res[, sd_feat_ratio   := sd_feat1 / sd_feat2]
  res[, lower_item_noise  := pmin(sd_feat1, sd_feat2)]
  res[, higher_item_noise := pmax(sd_feat1, sd_feat2)]
  res[sd_feat1 <= sd_feat2, bias_type := "lower_noise_bias"]
  res[sd_feat1 >  sd_feat2, bias_type := "higher_noise_bias"]
  res
}


simulate_equal_noise <- function(sd_feat_range         = seq(10, 60, 10),
                                 sd_spat               = 42,
                                 n_samples             = 20,
                                 use_nn_surfaces       = TRUE,
                                 averaged_surfaces_dir = NULL,
                                 work_dir              = getwd()) {
  par_grid <- expand.grid(sd_feat1 = sd_feat_range, sd_feat2 = sd_feat_range, sd_spat = sd_spat)
  par_grid <- par_grid[par_grid$sd_feat1 == par_grid$sd_feat2, ]
  res <- simulate_surfaces(par_grid,
                           n_samples             = n_samples,
                           skip_motor_noise      = TRUE,
                           use_nn_surfaces       = use_nn_surfaces,
                           averaged_surfaces_dir = averaged_surfaces_dir,
                           work_dir              = work_dir)
  setDT(res)
  res[, sd_feat_ratio   := sd_feat1 / sd_feat2]
  res[, lower_item_noise  := sd_feat1]
  res[, higher_item_noise := sd_feat2]
  res[sd_feat1 <= sd_feat2, bias_type := "lower_noise_bias"]
  res[sd_feat1 >  sd_feat2, bias_type := "higher_noise_bias"]
  res
}
