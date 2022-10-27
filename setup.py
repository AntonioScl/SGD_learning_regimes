from setuptools import setup, find_packages

setup(
    name="edm",
    packages=find_packages(),
    install_requires=[
        "jax",
        "dm-haiku",
    ],
)
