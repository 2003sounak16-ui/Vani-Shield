"""
Root re-export and CLI entrypoint for ml_pipeline/build_robust_manifest.py.
"""
import os
import sys

_curr = os.path.dirname(os.path.abspath(__file__))
_ml_dir = os.path.join(_curr, "ml_pipeline")
if _ml_dir not in sys.path:
    sys.path.insert(0, _ml_dir)

from ml_pipeline.build_robust_manifest import *

if __name__ == "__main__":
    main()
