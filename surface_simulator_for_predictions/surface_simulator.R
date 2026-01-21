#' Surface Simulator R Interface
#' 
#' This R script provides an interface to simulate surfaces using the Python 
#' GridBasedMultiConditionOptimizer through WSL. It creates parameter files,
#' runs Python simulation, and loads the results back into R.

library(arrow)
library(stringr)
library(data.table)

#' Simulate surfaces for given parameter combinations
#'
#' @param parameters Data frame with columns: sd_feat1, sd_feat2, sd_spat, condition_id
#'                  (sd_motor optional - will be set to 0 if skip_motor_noise=TRUE)
#' @param n_samples Number of samples used for training (determines which model to use)
#' @param skip_motor_noise Whether to skip motor noise computation (default: FALSE)
#' @param use_nn_surfaces Whether to use NN predictions (TRUE) or averaged surfaces from files (FALSE) (default: TRUE)
#'                       Note: NN surfaces only provide mu1 bias curves. For mu2 bias curves, use averaged surfaces (FALSE)
#' @param cleanup Whether to clean up temporary files (default: TRUE)
#' @return Data frame containing simulation results with columns: sd_feat1, sd_feat2, sd_spat, sd_motor, 
#'         feat_diff, mu1_density_asymmetry, mu2_density_asymmetry (if available), mu1_expectation, 
#'         mu2_expectation (if available), sd_curve
simulate_surfaces <- function(parameters, 
                             n_samples,
                             skip_motor_noise = FALSE,
                             use_nn_surfaces = TRUE,
                             cleanup = TRUE) {
  
  # Hardcoded paths
  win_path_to_wsl <- function(path) {
    stringr::str_replace( path, "(C|c):", "/mnt/c")
  }
  work_dir = win_path_to_wsl(getwd())
  print(work_dir)
    
  # Validate inputs
  if (skip_motor_noise) {
    required_cols <- c("sd_feat1", "sd_feat2", "sd_spat")
  } else {
    required_cols <- c("sd_feat1", "sd_feat2", "sd_spat", "sd_motor")
  }
  
  missing_cols <- setdiff(required_cols, names(parameters))
  if (length(missing_cols) > 0) {
    stop("Missing required columns in parameters: ", paste(missing_cols, collapse = ", "))
  }
  
  # Generate unique file names for this session
  session_id <- format(Sys.time(), "%Y%m%d_%H%M%S")
  input_file_name <- paste0("params_", session_id, ".arrow")
  output_file_name <- paste0("results_", session_id, ".arrow")
  
  # Convert WSL paths to Windows paths for R file operations
  input_file_win <- gsub("^/mnt/c", "C:", file.path(work_dir, input_file_name))
  output_file_win <- gsub("^/mnt/c", "C:", file.path(work_dir, output_file_name))
  
  cat("Creating parameter file:", input_file_win, "\n")
  
  # Write parameters to Arrow file
  write_parquet(parameters, input_file_win)
  
  # Build Python command arguments (just file names since we'll cd to work_dir)
  cmd_args <- c(
    input_file_name,
    n_samples,
    output_file_name
  )
  
  if (skip_motor_noise) {
    cmd_args <- c(cmd_args, "--skip-motor-noise")
  }
  
  if (!use_nn_surfaces) {
    cmd_args <- c(cmd_args, "--use-nn-surfaces")
  }
  
  # Create full command with cd and quoted paths
  full_cmd <- paste(
    "source ~/jax_env/bin/activate && cd", 
    shQuote(work_dir),
    "&& python surface_simulator.py", 
    paste(cmd_args, collapse = " ")
  )
  
  cat("Running simulation...\n")
  cat("Command:", full_cmd, "\n")
  
  # Execute command through WSL
  result <- shell(
    shell = 'wsl',
    cmd = full_cmd,
    flag = '',
    intern = TRUE
  )
  
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
  if (!file.exists(output_file_win)) {
    stop("Output file not found: ", output_file_win)
  }
  
  cat("Loading results from:", output_file_win, "\n")
  
  # Load Arrow results
  tryCatch({
    sim_results <- read_parquet(output_file_win)
    
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
      unlink(output_file_win)
      unlink(input_file_win)
      cat("Cleaned up temporary files\n")
    }
    
    return(long_results)
    
  }, error = function(e) {
    stop("Failed to load results: ", e$message)
  })
}

simulate_unequal_noise2 <- function(sd_feat_range = seq(10,60, 10), sd_spat = 42, n_samples = 20, use_nn_surfaces = TRUE){
  par_grid <- expand.grid(sd_feat1= sd_feat_range, sd_feat2 = sd_feat_range, sd_spat = sd_spat)
  res <- simulate_surfaces(par_grid,
                           n_samples = n_samples,
                           skip_motor_noise = TRUE,
                           use_nn_surfaces = use_nn_surfaces)
  setDT(res)
  res[,sd_feat_ratio:=sd_feat1/sd_feat2]
  res[,lower_item_noise:=pmin(sd_feat1, sd_feat2)]
  res[,higher_item_noise:=pmax(sd_feat1, sd_feat2)]
  res[sd_feat1<=sd_feat2, bias_type:='lower_noise_bias']
  res[sd_feat1>sd_feat2, bias_type:='higher_noise_bias']
  
  res
}



simulate_equal_noise <- function(sd_feat_range = seq(10,60, 10), sd_spat = 42, n_samples = 20, use_nn_surfaces = TRUE){
  par_grid <- expand.grid(sd_feat1 = sd_feat_range, sd_feat2 = sd_feat_range, sd_spat = sd_spat)
  par_grid <- par_grid[par_grid$sd_feat1==par_grid$sd_feat2,]
  res <- simulate_surfaces(par_grid,
                           n_samples = n_samples,
                           skip_motor_noise = TRUE,
                           use_nn_surfaces = use_nn_surfaces)
  setDT(res)
  res[,sd_feat_ratio:=sd_feat1/sd_feat2]
  res[,lower_item_noise:=sd_feat1]
  res[,higher_item_noise:=sd_feat2]
  res[sd_feat1<=sd_feat2, bias_type:='lower_noise_bias']
  res[sd_feat1>sd_feat2, bias_type:='higher_noise_bias']
  
  res
}