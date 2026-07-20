import unittest


class ImportTest(unittest.TestCase):
    def test_matrix_import_does_not_import_data_module(self):
        import sys

        sys.modules.pop("thor.data", None)
        from thor.utils.matrix import ld_entry

        self.assertTrue(callable(ld_entry))
        self.assertNotIn("thor.data", sys.modules)


if __name__ == "__main__":
    unittest.main()
