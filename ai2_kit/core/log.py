import logging

logging.basicConfig(level=logging.INFO)

def get_logger(name=None):
    return logging.getLogger(name)