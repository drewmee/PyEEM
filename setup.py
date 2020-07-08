import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="py-eema",
    version="0.0.1",
    author="Drew Meyers",
    author_email="drewm@mit.edu",
    description="A description",
    long_description="A longer description",
    long_description_content_type="text/markdown",
    url="https://github.com/drewmee/PyEEMA",
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[],

)