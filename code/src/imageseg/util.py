#Generic utility functions such as plotting, or loading/saving parameter files.

import json

def save_param(paramdict,paramfile):
    """saves parameter dict to JSON"""
    with open(paramfile,'rb') as f:
        json.save(paramdict,f)

def load_param(paramfile):
    """loads parameters from JSON to a Python Dict"""
    with open(paramfile,'rb') as f:
        param_dict=json.load(f)
    return param_dict
