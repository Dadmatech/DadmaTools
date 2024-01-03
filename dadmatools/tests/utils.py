import sys
import os

def set_root_dir():
    current_path = os.path.abspath(os.getcwd())
    # adding Folder_2/subfolder to the system path
    sys.path.insert(0, current_path[:-len('/dadmatools/tests')])