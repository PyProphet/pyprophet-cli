# encoding: utf-8
from __future__ import print_function

import click

Path = click.Path
option = click.option

job_number = option("--job-number", default=1, help="1 <= job-number <= job-count [default=1]")
job_count = option("--job-count", default=1, help="overall number of batch jobs [default=1]")
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

chunk_size = option("--chunk-size", default=100000,
                    help="chunk size when reading input files, may impact processing speed "
                    "[default=100000]",
                    )

extra_group_columns = option("--extra-group-column", "extra_group_columns", type=str, multiple=True,
                            help="additionally compute score over this group, you may repeat this option")

data_filename_pattern = option("--data-filename-pattern", default="*.txt",
                               help="glob pattern to filter files in data folder")

ignore_invalid_scores = option("--ignore-invalid-scores", is_flag=True,
                               help="ignore columns where all values are invalid/missing")


def transform_sep(ctx, param, value):
    return {"tab": "\t", "comma": ",", "semicolon": ";"}.get(value)


separator = option("--separator", type=click.Choice(("tab", "comma", "semicolon")),
                   default="tab", help="separator for input and output files [default=tab]",
                   callback=transform_sep)

sample_factor = option("--sample-factor", required=True, type=float,
                       help="sample factor in range 0.0 .. 1.0")
