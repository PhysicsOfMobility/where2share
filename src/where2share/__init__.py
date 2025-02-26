__author__ = """Felix Jung & Philip Marszal"""
__email__ = "felix.jung@tu-dresden.de"
__version__ = "0.1.0"

import logging

logging.getLogger("").setLevel("INFO")

formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(name)s: %(message)s")

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logging.getLogger("").addHandler(ch)

# read config once
# config = read_config()
#
# fh = logging.FileHandler(filename=config.logfile_path, encoding="utf-8", mode="a")
# fh.setFormatter(formatter)
# logging.getLogger("").addHandler(fh)

# logging.getLogger("").setLevel(config.log_level)
