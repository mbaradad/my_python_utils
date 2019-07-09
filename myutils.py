from .common_utils import *
import sys
if sys.version_info >= (3,0):
  from .myutils3 import *
else:
  from .myutils2 import *