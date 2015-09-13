# encoding: utf-8
# vi: et sw=4 ts=4
from __future__ import print_function

import pkg_resources
from exceptions import WorkflowError


class JobMeta(type):

    def __new__(cls_, name, parents, dd):
        to_check = "command_name", "options", "run"
        if object not in parents:
            if any(field not in dd for field in to_check):
                raise TypeError("needed attributes/methods: %s" % ", ".join(to_check))
        return super(JobMeta, cls_).__new__(cls_, name, parents, dd)


class Job(object):

    __metaclass__ = JobMeta


def _load_drivers():

    for ep in pkg_resources.iter_entry_points("pyprophet_cli_plugin", name="config"):
        try:
            driver = ep.load()
        except Exception:
            raise
            raise WorkflowError("driver %s can not be loaded" % ep)
        try:
            name, options, run, help_ = driver()
        except Exception:
            raise
            raise WorkflowError("driver %s can not be loaded" % ep)
        yield _create_run_job(name, options, run, help_)


def _create_run_job(name, options, job_function, help_):

    class _RunWorkflow(Job):
        options = command_name = None

        def run(self):
            job_function(self)

    _RunWorkflow.options = options
    _RunWorkflow.command_name = name
    _RunWorkflow.__doc__ = help_
    _RunWorkflow.run = job_function
    return _RunWorkflow


for driver in _load_drivers():
    pass
