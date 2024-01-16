from setuptools import setup, find_packages

setup(
    name="trajectories_to_forces",
    version="1.0.0",
    author="Maarten Bransen",
    license='MIT License',
    long_description=open('README.md').read(),
    packages=find_packages(include=["trajectories_to_forces", "trajectories_to_forces.*"]),
    install_requires=[
        "numpy>=1.19.2",
        "scipy>=1.6.0",
        "pandas>=1.2.0",
        "numba>=0.58.0",
    ],
)
