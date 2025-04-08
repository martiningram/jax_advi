from os import getenv
from setuptools import setup
from setuptools import find_packages


setup(
    name="jax-advi",
    use_scm_version={"version_scheme": "no-guess-dev"},
    description="ADVI in JAX a la Giordano et al",
    packages=find_packages(),
)
