import os

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.client import Config
from tqdm import tqdm


def _get_demo_dataset_info():
    demos = [
        {
            "demo_name": "rutherford",
            "description": "Excitation Emission Matrix (EEM) fluorescence spectra used for "
            "combustion generated particulate matter source identification using a neural network.",
            "citation": 'Rutherford, Jay W., et al. "Excitation emission matrix fluorescence '
            'spectroscopy for combustion generated particulate matter source identification." '
            "Atmospheric Environment 220 (2020): 117065.",
            "DOI": "10.1016/j.atmosenv.2019.117065",
            "absorbance_instrument": "Aqualog",
            "water_raman_instrument": None,
            "EEM_instrument": "Aqualog",
        },
        {
            "demo_name": "drEEM",
            "description": "The demo dataset contains measurements made during four "
            "surveys of San Francisco Bay that took place in spring, summer, autumn "
            "and winter 2006 (Murphy et al. 2013, J. Mar. Syst. 111-112, 157-166).",
            "citation": 'Murphy, Kathleen R., et al. "Fluorescence spectroscopy and '
            'multi-way techniques. PARAFAC." Analytical Methods 5.23 (2013): 6557-6566.',
            "DOI": "10.1039/c3ay41160e",
            "absorbance_instrument": "Cary 4E",
            "water_raman_instrument": "Fluorolog",
            "EEM_instrument": "Fluorolog",
        },
    ]

    return pd.DataFrame.from_records(demos)


# from pathlib import Path
# Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)


def _get_bucket_file_list(bucket_name, bucket_dir):
    s3_resource = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
    bucket = s3_resource.Bucket(bucket_name)
    return [i.key for i in bucket.objects.filter(Prefix=bucket_dir)]


def _download_S3_dir(demo_data_dir, bucket_dir, overwrite):
    # TODO consider changing to zip files for each demo dataset
    # Download zip file and unzip instead of downloading
    # each file one by one.
    bucket_name = "pyeem-demo-datasets"
    s3_resource = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
    bucket = s3_resource.Bucket(bucket_name)

    file_list = [i.key for i in bucket.objects.filter(Prefix=bucket_dir)]
    for f in tqdm(file_list, desc="Download Demo Dataset from S3"):
        path = os.path.join(demo_data_dir, f)
        if os.path.exists(path) and overwrite == False:
            continue

        if os.path.normpath(f) == bucket_dir:
            continue

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        bucket.download_file(f, path)


def download_demo(data_dir, demo_name, overwrite=False):
    """Download a demo dataset from AWS S3. Please note that this step requires
    an internet connection because the data is downloaded from an AWS S3 bucket.

    Args:
        data_dir (str): The directory in which the demo data will be downloaded.
        demo_name (str): The name of the demo dataset you would like to download.
        overwrite (bool, optional): Determines whether or not the demo directory
            will be overwritten if it already exists. Defaults to False.

    Returns:
        str: The relative path to the demo directory.
    """
    demos = _get_demo_dataset_info()
    if demo_name not in demos["demo_name"]:
        ValueError("%s does not exist in demos['demo_name']" % demo_name)

    abs_data_dir = os.path.abspath(data_dir)
    _download_S3_dir(abs_data_dir, demo_name, overwrite)

    demo_dir = os.path.join(abs_data_dir, demo_name)
    rel_demo_dir = os.path.relpath(demo_dir, os.getcwd())
    return rel_demo_dir
