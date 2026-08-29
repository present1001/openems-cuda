# -*- coding: utf-8 -*-
#
# Shortcut openEMS import
from __future__ import absolute_import

import os
import ctypes

# Python 3.8+ no longer searches PATH for dependent DLLs of extension modules.
# CSXCAD registers its own directory below; register a separate openEMS prefix
# as well so a CUDA build can coexist with the normal dependency installation.
_openems_dll_directory = None
_cuda_dll_directory = None
_openems_library = None
if os.name == 'nt':
    _install_path = os.environ.get('OPENEMS_INSTALL_PATH', None)
    if _install_path and os.path.exists(_install_path):
        try:
            _openems_dll_directory = os.add_dll_directory(_install_path)
        except AttributeError:  # Python < 3.8
            if _install_path not in os.environ['PATH']:
                os.environ['PATH'] += os.pathsep + _install_path

    _cuda_path = os.environ.get('CUDA_PATH', None)
    if _cuda_path:
        _cuda_bin = os.path.join(_cuda_path, 'bin')
        if os.path.exists(_cuda_bin):
            try:
                _cuda_dll_directory = os.add_dll_directory(_cuda_bin)
            except AttributeError:  # Python < 3.8
                if _cuda_bin not in os.environ['PATH']:
                    os.environ['PATH'] += os.pathsep + _cuda_bin

# try to import CSXCAD, e.g. make sure the (windows) dll path are set
import CSXCAD

# CSXCAD may be installed in a directory that also contains a CPU-only
# openEMS.dll. Preload the explicitly selected library by absolute path so the
# extension module cannot accidentally bind to the other copy.
if os.name == 'nt' and _install_path:
    _library_path = os.path.join(_install_path, 'openEMS.dll')
    if os.path.isfile(_library_path):
        _openems_library = ctypes.WinDLL(_library_path)

try:
    from importlib.metadata import version
    __version__ = version("openEMS")
except ImportError:
    try:
        from openEMS.__fallback_version__ import __fallback_version__
        __version__ = __fallback_version__
    except ImportError:
        __version__ = "0.0.0"

from openEMS.openEMS import openEMS
