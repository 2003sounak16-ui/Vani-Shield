"""
Root re-export of ml_pipeline/build_robust_manifest.py for backward compatibility.
"""
import os
import sys

_curr = os.path.dirname(os.path.abspath(__file__))
_ml_dir = os.path.join(_curr, "ml_pipeline")
if _ml_dir not in sys.path:
    sys.path.insert(0, _ml_dir)

from ml_pipeline.build_robust_manifest import *
