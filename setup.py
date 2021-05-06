from setuptools import setup, find_packages

setup(
    name="trajectories_to_forces",
    version="0.1.0",
    author="Maarten Bransen",
    author_email="m.bransen@uu.nl",
    license='GNU General Public License v3.0',
    long_description=open('README.md').read(),
    packages=find_packages(include=["trajectories_to_forces", "trajectories_to_forces.*"]),
    install_requires=[
        "numpy>=1.19.2",
        "scipy>=1.6.0",
        "pandas>=1.2.0",
    ],
)
