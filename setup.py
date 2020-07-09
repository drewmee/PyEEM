import setuptools

__version__ = "1.0.0"

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyeem",
    version=__version__,
    author="Drew Meyers",
    author_email="drewm@mit.edu",
    description="A description",
    long_description="A longer description",
    long_description_content_type="text/markdown",
    url="https://github.com/drewmee/PyEEM",
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    test_suite='tests',
    python_requires='>=3.6',
    install_requires=[
        'tensorflow',
        'pandas',
        'h5py',
        'tables',
        'matplotlib',
        'tensorly',
        'keras',
    ],

)