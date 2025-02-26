import os

import numpy as np

from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    ext_modules=cythonize(
        [
            Extension(
                name="*",
                sources=["src/where2share/**/*.pyx"],
                include_dirs=[np.get_include()],
            ),
        ],
        compiler_directives={"embedsignature": True},
    ),
    options={
        "build_ext": {
            "inplace": True,
            "parallel": os.cpu_count() - 1,
        }
    },
)
