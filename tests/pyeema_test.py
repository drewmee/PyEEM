import unittest
import pyeema
import os


class TestLoad(unittest.TestCase):
    def testInsufficientArgs(self):
        data_dir = None
        self.assertRaises(Exception, pyeema.Load, data_dir)

    def testNonExistentDataDirPath(self):
        data_dir = "some non-existent path"
        self.assertRaises(FileNotFoundError, pyeema.Load, data_dir)

    def testValidDataDirPath(self):
        try:
            data_dir = "data/mock/"
            pyeema.Load(data_dir)
        except Exception:
            self.fail("Load() raised Exception unexpectedly!")

    def testHdf5Creation(self):
        data_dir = "data/mock/"
        load = pyeema.Load(data_dir)
        required_subdirs = ['corrections', 'processed', 'raw_sample_sets']
        assert set(list(load.hdf5_root.keys())) == set(required_subdirs)

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


class Testpreprocessing(unittest.TestCase):
    def testCrop(self):
        assert 1 == 1


if __name__ == '__main__':
    unittest.main()
