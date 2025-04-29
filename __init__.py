from .coords import *
from .SL import *
from .cone import *
from .movie import *
from .animation import *
from .heatmaps import get_heatmaps
from .h5 import *
from .file import *
import importlib
import os

# sample comment
def six_to_four_format(coords):
    coords = coords.reshape(6,2)
    coords = coords[[0,1,3,5]]
    return coords


class EchoTools:
    """Singleton class to import the parent module dynamically."""
    _instance = None

    def __new__(cls, cfg=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            absolute_path = os.path.abspath(__file__)
            cwd = os.getcwd()
            parent_dir = os.path.dirname(os.path.relpath(absolute_path, cwd))
            parent_module_name = '.'.join(parent_dir.split(os.sep))
            cls._instance.parent_module = importlib.import_module(parent_module_name)
        return cls._instance

    def get_instance(self):
        """Get the dynamically imported parent module."""
        return self.parent_module
        
