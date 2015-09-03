# encoding: utf-8

# vim: et sw=4 ts=4

import logging
import math
import random
import string
import time

import click

@click.group()
def cli():
    pass

from core import Job

levels = map(string.lower, "CRITICAL ERROR WARNING INFO DEBUG".split())

for sub_class in Job.__subclasses__():

    def create_handler(sub_class=sub_class):

        @click.option("--log-level", type=click.Choice(levels), default="debug",
                      help="[default=debug]")
        @click.option("--log-file", type=click.File(mode="w"),
                      help="write logs to given file instead to stderr")
        def handler(log_level, log_file, **options):
            logger = logging.getLogger("pyprophet-jobs")
            logger.setLevel(log_level.upper())
            h = logging.StreamHandler(stream=log_file)
            fmt = "%(levelname)-8s -- [pid=%(process)5d] : %(asctime)s: %(message)s"
            h.setFormatter(logging.Formatter(fmt))
            logger.addHandler(h)
            options["logger"] = logger

            for (key, value) in options.items():
                logger.info("got setting %s=%r" % (key, value))

            inst = sub_class()
            # set instance attributes according to command line options:
            inst.__dict__.update(options)
            inst.run()
        handler.__doc__ = getattr(sub_class, "__doc__", "")
        return handler

    handler = create_handler()
    # we set the options in reverse order, so they will appear in the right order
    # if click printw the help message:
    for option in sub_class.options[::-1]:
        handler = option(handler)

    command_name = sub_class.command_name
    cli.command(name=command_name)(handler)



"""
todo: resolve order of jobs and create the "super command"
"""

