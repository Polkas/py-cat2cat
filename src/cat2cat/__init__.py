# read version from installed package
from importlib.metadata import version

__version__ = version("cat2cat")

# simplified
from cat2cat.cat2cat import cat2cat

from cat2cat.cat2cat_ml import cat2cat_ml_run
