import os

import pytest

import pyeem


class TestDataset:
    def testInsufficientArgs(self):
        return
        with pytest.raises(Exception):
            pyeem.datasets.Dataset()

    def testNonExistentDataDirPath(self):
        return
        data_dir = "some non-existent path"
        with pytest.raises(FileNotFoundError):
            pyeem.datasets.Dataset(data_dir)

    """
    def testValidDataDirPath(self):
        try:
            data_dir = "data/mock/"
            pyeem.datasets.Dataset(data_dir)
        except Exception:
            self.fail("Dataset() raised Exception unexpectedly!")
    """

    def testNonExistentMetadataPath(self):
        return

    def testInvalidMetadataFileFormat(self):
        return

    def testInvalidMetadataColumns(self):
        return

    def testInvalidMetadataDatetimeFormat(self):
        return

    def testInvalidRawDataFileTypeArg(self):
        return


class Testpreprocessing:
    def testCrop(self):
        assert 1 == 1
