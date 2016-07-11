# About

``pyprophet-cli`` is a collection of command line tools which can be chained to run the new ``pyprophet`` workflow either locally or on ETH super computer brutus.


## Command Line Tool

After instalation wiht ``pip`` you can get more help using
````
$ pyprophet-cli --help

    Usage: pyprophet-cli [OPTIONS] COMMAND [ARGS]...

    Options:
    --help  Show this message and exit.

    Commands:
    apply_weights  applies weights to given data files and...
    check          runs validity tests on input files in...
    learn          runs pyprophet learner on data files in...
    score          applies weights to given data files and...
    subsample      subsamples transition groups from given input...

````

## General Workflow

1. ``check``:  check n input files in parallel

2. ``subsample``:  subsample n input files, create n output files which are smaller by a factor eg 20

3. ``learn``: read all subsampled input files into memory and run pyprophet learner.
   -> computes weights for basic scores and writes them to disc

4. ``apply_weights``:  apply weights from step 3 to n input files in parallel,
   writes n files with computed "classifier scores"

5. ``score``:  read all classifier scores from 4., use statistics to compute parameters of final
   "q-score" decorate n input files in parallel with the new "q-score" column

