import logging

# format to include timestamp and module
logging.basicConfig(format='%(asctime)s %(name)s: %(message)s', level=logging.INFO)
# suppress transitions logging
logging.getLogger('transitions.core').setLevel(logging.WARNING)

def get_logger(name=None):
    return logging.getLogger(name)