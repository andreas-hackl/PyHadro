import pyhadro.np_json
import pyhadro.db
import pyhadro.gevp
import pyhadro.gpt_io 

from pyhadro.db import Database


import time

class Timer():
    def __init__(self):
        self.start = time.time()
    
    def __call__(self):
        return time.time() - self.start
    
    def __str__(self): 
        return f"LQCDPY {self.__call__():20.5f} sec    :   "
    
timer = Timer()

def message(s, *args):
    print(f"{timer.__str__()}{s}", *args, flush=True)

