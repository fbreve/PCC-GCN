# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 16:23:28 2026

@author: fbrev
"""

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(
        "lnpcc_step.pyx",
        compiler_directives={'language_level': 3}  # Py3
    ),
    include_dirs=[np.get_include()]  # NumPy
)