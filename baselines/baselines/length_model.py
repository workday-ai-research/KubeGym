"""Shim so the vendored source `controllers.py` imports unmodified.

The source file does `from length_model import LengthModel, load_length_model`
after inserting its own directory on `sys.path`.  `kubegym` vendors that module
verbatim (INTERFACE.md section 10, sha256 7d35ce5e...60d34), so this re-exports
it rather than forking a second copy.
"""
from kubegym.models.length_model import *          # noqa: F401,F403
from kubegym.models.length_model import LengthModel, load_length_model  # noqa: F401
