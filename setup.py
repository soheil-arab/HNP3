from distutils.core import setup

from Cython.Build import cythonize
from setuptools import Extension
import numpy

extension = Extension("common.helper",
                      ["common/helper.pyx"],
                      include_dirs=[numpy.get_include()])

setup(
    name="HNP3",
    version="1.0",
    packages=['Generator', 'HNP3', ''],
    ext_modules=cythonize([extension]), requires=['scipy', 'numpy']
)
