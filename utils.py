# coding: utf-8
"""
utility functions
"""

from logging import getLogger, StreamHandler

__author__ = "nyk510"


def get_logger(log_level="DEBUG"):
    logger = getLogger(__name__)
    logger.setLevel(log_level)
    handler = StreamHandler()
    handler.setLevel(log_level)
    logger.addHandler(handler)
    return logger
