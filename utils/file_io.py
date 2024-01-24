# Description: File IO utils
# Borrowed from [VPT](https://github.com/KMnP/vpt). Thanks to the authors.

from iopath.common.file_io import PathManager as PathManagerBase
from iopath.common.file_io import HTTPURLHandler


PathManager = PathManagerBase()
PathManager.register_handler(HTTPURLHandler())
