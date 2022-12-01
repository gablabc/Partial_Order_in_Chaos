from setuptools import setup, Extension

# Compile *mysum.cpp* into a shared library 
setup(
    #...
    ext_modules=[Extension('treeshap', sources=['uxai/tree_shap/main.cpp'],),],
)