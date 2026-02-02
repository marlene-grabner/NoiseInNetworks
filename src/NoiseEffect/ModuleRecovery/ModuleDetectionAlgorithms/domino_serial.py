# File: domino_serial.py
import sys
import multiprocessing

# --- PATCH 1: Fix Parallel Processing Crash (Apple Silicon) ---
class SerialPool:
    """A dummy class that replaces multiprocessing.Pool"""
    def __init__(self, processes=None): pass
    def map(self, func, iterable): return list(map(func, iterable))
    def close(self): pass
    def join(self): pass

# --- PATCH 2: Disable Broken Visualization ---
def dummy_visualize(*args, **kwargs):
    print("[INFO] Skipping visualization (Incompatible with custom IDs)")
    return

try:
    # 1. Apply Parallel Patch
    import src.core.domino
    src.core.domino.multiprocessing.Pool = SerialPool
    
    # 2. Apply Visualization Patch
    # We overwrite the visualization function with a dummy function that does nothing
    import src.runner
    src.runner.visualize_modules = dummy_visualize
    
    print("[INFO] Successfully patched DOMINO (Serial Mode + No Vis).")

except ImportError:
    print("[ERROR] Could not find src.core.domino. Make sure you are in the domino-env!")
    sys.exit(1)

# --- RUN THE ORIGINAL PROGRAM ---
from src.runner import main_domino

if __name__ == "__main__":
    sys.exit(main_domino())
