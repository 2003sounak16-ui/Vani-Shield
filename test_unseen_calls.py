"""
VaniShield AI — Root Entry Point for Unseen Calls Generalization Test
Forwards execution to ml_pipeline/test_unseen_calls.py.
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(SCRIPT_DIR, "ml_pipeline", "test_unseen_calls.py")

if __name__ == "__main__":
    if os.path.exists(TARGET):
        # Execute target module
        with open(TARGET, "r", encoding="utf-8") as f:
            code = compile(f.read(), TARGET, "exec")
            exec(code, {"__name__": "__main__", "__file__": TARGET})
    else:
        print(f"[ERROR] Target script not found at {TARGET}")
