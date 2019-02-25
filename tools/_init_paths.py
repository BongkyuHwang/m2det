import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

cur_dir = os.path.dirname(__file__)
lib_path = os.path.join(cur_dir, "..", "lib")
add_path(lib_path)
