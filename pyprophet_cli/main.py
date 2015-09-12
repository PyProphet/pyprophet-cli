# encoding: utf-8
# vim: et sw=4 ts=4
from __future__ import print_function

import logging
import string

import click

from version import version

# we import all subcommands here so that the subclases of Job are registered below !
import prepare
import subsample
import learn
import apply_weights
import score

from core import Job


def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo("%d.%d.%d" % version)
    ctx.exit()


@click.group()
@click.option("--version", is_flag=True, callback=print_version, expose_value=False, is_eager=True,
              help="print version of pyprophet-cli")
def cli():
    pass


levels = map(string.lower, "CRITICAL ERROR WARNING INFO DEBUG".split())

for sub_class in Job.__subclasses__():

    def create_handler(sub_class=sub_class):

        @click.option("--log-level", type=click.Choice(levels), default="debug",
                      help="[default=debug]")
        @click.option("--log-file", type=click.File(mode="w"),
                      help="write logs to given file instead to stderr")
        def handler(log_level, log_file, **options):
            logger = logging.getLogger("pyprophet-cli")
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
