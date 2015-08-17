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

    def create(sub_class=sub_class):
        def handler(log_level, log_file, **options):
            logger = logging.getLogger("pyprophet-brutus")
            logger.setLevel(log_level.upper())
            h = logging.StreamHandler(stream=log_file)
            fmt = "%(levelname)-8s -- [pid=%(process)5d] : %(asctime)s: %(message)s"
            h.setFormatter(logging.Formatter(fmt))
            logger.addHandler(h)
            options["logger"] = logger
            sub_class().run(**options)
        return handler

    handler = click.option("--log-level", type=click.Choice(levels), default="debug")(create())
    handler = click.option("--log-file", type=click.File(mode="w"))(handler)
    for option in sub_class.options:
        handler = option(handler)

    command_name = sub_class.command_name
    cli.command(name=command_name)(handler)



"""
todo: resolve order of jobs and create the "super command"
"""

