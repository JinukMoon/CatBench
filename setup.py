"""Setup configuration for CatBench package."""

from setuptools import setup, find_packages

setup(
    name="catbench",
    version='1.0.0',
    author="JinukMoon",
    author_email="jumoon@snu.ac.kr",
    packages=find_packages(),
    package_data={
        'catbench.dispersion': [
            'cuda/*.py',
            'cuda/*.so',
            'cuda/pair_e3gnn/*.cu',
            'cuda/pair_e3gnn/*.h',
        ],
    },
    description="CatBench: Benchmark Framework of Machine Learning Interatomic Potentials for Adsorption Energy Predictions in Heterogeneous Catalysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JinukMoon/catbench",
    license="MIT",
    install_requires=[
        "ase>=3.22.1",
        "numpy>=1.20.0",
        "pandas>=1.3.0", 
        "matplotlib>=3.4.0",
        "requests>=2.25.0",
        "xlsxwriter>=3.2.0",
        "pyyaml>=5.4.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "d3": [
            "torch>=1.12.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="MLIP benchmark for catalysis",
    python_requires=">=3.8",
)