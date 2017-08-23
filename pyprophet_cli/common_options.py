# encoding: utf-8
from __future__ import print_function

import sys
import click
import numpy as np

Path = click.Path
option = click.option

job_number = option("--job-number", default=1, show_default=True, help="1 <= job-number <= job-count")
job_count = option("--job-count", default=1, show_default=True, help="overall number of batch jobs")
data_folder = option("--data-folder", help="folder of input data to process",
                     type=Path(exists=True, file_okay=False, dir_okay=True, readable=True),
                     required=True)

work_folder = option("--work-folder",
                     help="folder for intermediate results which are needed by following processing steps",
                     type=Path(file_okay=False, dir_okay=True, readable=True, writable=True),
                     required=True)

result_folder = option("--result-folder",
                       help="folder for final results",
                       type=Path(file_okay=False, dir_okay=True, writable=True),
                       required=True)

local_folder = option("--local-folder",
                      type=Path(file_okay=False, dir_okay=True, readable=True, writable=True),
                      help="local folder on computing node ($TMPDIR for brutus lsf)")

random_seed = option("--random-seed", type=int,
                     help="set a fixed seed for reproducable results")

chunk_size = option("--chunk-size", default=100000, show_default=True,
                    help="chunk size when reading input files, may impact processing speed ",
                    )

def transform_lambda_(ctx, param, value):
  if value[1] == 0 and value[2] == 0:
      lambda_ = value[0]
  elif 0 <= value[0] < 1 and value[0] <= value[1] <= 1 and 0 < value[2] < 1:
      lambda_ = np.arange(value[0], value[1], value[2])
  else:
      sys.exit('Error: Wrong input values for pi0_lambda. pi0_lambda must be within [0,1).')
  return(lambda_)

lambda_ = option('--lambda', "lambda_", default=[0.4,0,0], show_default=True, type=(float, float, float), help='Use non-parametric estimation of p-values. Either use <START END STEPS>, e.g. 0.1, 1.0, 0.1 or set to fixed value, e.g. 0.4.', callback=transform_lambda_)

pi0_method = option('--pi0_method', default='smoother', show_default=True, type=click.Choice(['smoother', 'bootstrap']), help='Either "smoother" or "bootstrap"; the method for automatically choosing tuning parameter in the estimation of pi_0, the proportion of true null hypotheses.')

pi0_smooth_df = option('--pi0_smooth_df', default=3, show_default=True, type=int, help='Number of degrees-of-freedom to use when estimating pi_0 with a smoother.')

pi0_smooth_log_pi0 = option('--pi0_smooth_log_pi0/--no-pi0_smooth_log_pi0', default=False, show_default=True, help='If True and pi0_method = "smoother", pi0 will be estimated by applying a smoother to a scatterplot of log(pi0) estimates against the tuning parameter lambda.')

lfdr_truncate = option('--lfdr_truncate/--no-lfdr_truncate', default=True, show_default=True, help='If True, local FDR values >1 are set to 1.')

lfdr_monotone = option('--lfdr_monotone/--no-lfdr_monotone', default=True, show_default=True, help='If True, local FDR values are non-decreasing with increasing p-values.')

lfdr_transformation = option('--lfdr_transformation', default='probit', show_default=True, type=click.Choice(['probit', 'logit']), help='Either a "probit" or "logit" transformation is applied to the p-values so that a local FDR estimate can be formed that does not involve edge effects of the [0,1] interval in which the p-values lie.')

lfdr_adj = option('--lfdr_adj', default=1.5, show_default=True, type=float, help='Numeric value that is applied as a multiple of the smoothing bandwidth used in the density estimation.')

lfdr_eps = option('--lfdr_eps', default=np.power(10.0,-8), show_default=True, type=float, help='Numeric value that is threshold for the tails of the empirical p-value distribution.')

extra_group_columns = option("--extra-group-column", "extra_group_columns", type=str, multiple=True,
                            help="additionally compute score over this group, you may repeat this option")

data_filename_pattern = option("--data-filename-pattern", default="*.txt",
                               help="glob pattern to filter files in data folder")

ignore_invalid_scores = option("--ignore-invalid-scores", is_flag=True,
                               help="ignore columns where all values are invalid/missing")

statistics_mode = click.option("--statistics-mode", type=click.Choice(['run-specific', 'global', 'experiment-wide']))

def transform_sep(ctx, param, value):
    return {"tab": "\t", "comma": ",", "semicolon": ";"}.get(value)


separator = option("--separator", type=click.Choice(("tab", "comma", "semicolon")),
                   default="tab", help="separator for input and output files [default=tab]",
                   callback=transform_sep)

sample_factor = option("--sample-factor", required=True, type=float,
                       help="sample factor in range 0.0 .. 1.0")
