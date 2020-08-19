import setuptools

__version__ = "0.1.0"

with open("README.md", "r") as fh:
    long_description = fh.read()

# Make sure find_packages exclude is correct
# data_files needed?
setuptools.setup(
    name="pyeem",
    version=__version__,
    author="Drew Meyers",
    author_email="drewm@mit.edu",
    description="Python library for the preprocessing, correction, deconvolution and analysis of Excitation Emission Matrices (EEMs).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/drewmee/PyEEM",
    license="MIT",
    packages=setuptools.find_packages(exclude=["tests*", "paper", "docs"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    test_suite="tests",
    python_requires='>=3.6, <=3.8',
    install_requires=[
        "numpy<1.19.0,>=1.18.5",
        "pandas>=1.0.5",
        "h5py>=2.10.0",
        "tables>=3.6.1",
        "matplotlib>=3.3.0",
        "celluloid>=0.2.0",
        "docutils<0.16,>=0.10",
        "urllib3>=1.25.9",
        "boto3>=1.14.33",
        "tqdm>=4.48.0",
        "scipy==1.4.1",
        "tensorflow>=2.2.0",
    ],
    extras_require={
        "docs": [
            "sphinx>=3.2.0",
            "sphinx-automodapi>=0.12",
            "sphinx-rtd-theme>=0.5.0",
            "msmb_theme>=1.2.0",
            "nbsphinx>=0.7.1",
            "sphinx-copybutton>=0.3.0",
            "black>=18.9b0",
            "isort>=5.4.2",
            "rstcheck>=3.3.1",
        ],
        "tests": ["pytest>=6.0.1", "tox>=3.16.1"],
        "develop": ["twine>=3.2.0"],
        "jupyter": ["jupyter>=1.0.0", "jupyterlab>=2.2.2"],
    },
    package_data={"pyeem": ["plots/pyeem_base.mplstyle"]},
)
