import argparse
import mmap
import os
import re
import subprocess
from torch.utils.cpp_extension import CUDA_HOME

def run_cuobjdump(file_path):
    import platform
    exe_name = 'cuobjdump.exe' if platform.system() == 'Windows' else 'cuobjdump'
    cmd_path = os.path.join(CUDA_HOME, 'bin', exe_name)
    
    command = [cmd_path, '-sass', file_path]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"cuobjdump failed with error:\n{result.stderr}")
        
    return result.stdout