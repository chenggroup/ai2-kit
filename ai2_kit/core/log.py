import logging
import os

level_name = os.environ.get('LOG_LEVEL', 'INFO')
level = logging._nameToLevel.get(level_name, logging.INFO)

# format to include timestamp and module
logging.basicConfig(format='%(asctime)s %(name)s: %(message)s', level=level)
# suppress transitions logging
logging.getLogger('transitions.core').setLevel(logging.WARNING)

def get_logger(name=None):
    return logging.getLogger(name)