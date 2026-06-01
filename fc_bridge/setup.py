from setuptools import setup, find_packages

setup(
    name="fc_bridge",
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pymavlink",
    ],
)
