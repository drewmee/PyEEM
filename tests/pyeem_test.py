import os

import pytest

import pyeem


class TestLoad:
    def testInsufficientArgs(self):
        with pytest.raises(Exception):
            pyeem.datasets.Load()

    def testNonExistentDataDirPath(self):
        data_dir = "some non-existent path"
        with pytest.raises(FileNotFoundError):
            pyeem.datasets.Load(data_dir)

    """
    def testValidDataDirPath(self):
        try:
            data_dir = "data/mock/"
            pyeem.datasets.Load(data_dir)
        except Exception:
            self.fail("Load() raised Exception unexpectedly!")
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
