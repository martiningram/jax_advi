from os import getenv
from setuptools import setup
from setuptools import find_packages


setup(
    name="jax-advi",
    version=getenv("VERSION", "LOCAL"),
    description="ADVI in JAX a la Giordano et al",
    packages=find_packages(),
)
