#!/usr/bin/env Rscript

library(EpiNow2)

estim_R <- function(infpath, outfpath) {
  # no reporting delay 
  reporting_delay <- list(
    mean = 1, sd = 0, max = 1, dist = "gamma"
  )
  # sars-cov-2
  generation_time <- list(
    mean = 3.635, mean_sd = 0, sd = 3.075, sd_sd = 0., max = 10, dist = 'gamma'
  )
  incubation_period <- list(
    mean = 1.621, mean_sd = 0, sd = 0.418, sd_sd = 0., max = 10, dist = 'lognormal'
  )
  
  reported_cases <- read.csv(infpath)
  reported_cases$date <- as.Date(reported_cases$date)
  reported_cases$confirm <- as.numeric(reported_cases$confirm)
  
  estimates <- epinow(
    reported_cases = reported_cases,
    generation_time = generation_time_opts(generation_time),
    delays = delay_opts(incubation_period, reporting_delay),
    rt = rt_opts(prior = list(mean = 2, sd = 0.2)),
    stan = stan_opts(cores = 4, control = list(adapt_delta = 0.99)),
    verbose = interactive()
  )
  
  df <- summary(estimates, type = "parameters", params = "R")
  write.csv(df, outfpath)
}

args = commandArgs(trailingOnly = TRUE)
infpath = args[1]
outfpath = args[2]
estim_R(infpath, outfpath)
