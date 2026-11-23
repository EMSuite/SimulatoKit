"""Tests of tools/pins/compare.py on planted changes in a scratch git repository.

    python3 -I -B tools/pins/test_compare.py        (the reference job runs it before the generators)
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

COMPARE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare.py")


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = self.dir.name
        os.makedirs(os.path.join(self.root, "Tests/Fixtures"))
        self.write("Tests/Fixtures/pins.json", {"a": 0.1, "n": 3, "rows": [1.0, 2.5e-17], "python": "3.12.14",
                                                "checks": ["ok  far field: 1.110e-16 <= 1e-13"], "name": "Si"})
        for args in (["init", "-q"], ["add", "."], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"]):
            subprocess.run(["git", *args], cwd=self.root, check=True)

    def tearDown(self):
        self.dir.cleanup()

    def write(self, path, value, raw=None):
        with open(os.path.join(self.root, path), "wb") as f:
            f.write(raw if raw is not None else json.dumps(value).encode())

    def run_compare(self):
        return subprocess.run([sys.executable, "-I", "-B", COMPARE], cwd=self.root, capture_output=True, text=True)

    def test_no_change_passes(self):
        self.assertEqual(self.run_compare().returncode, 0)

    def test_float_noise_passes(self):
        self.write("Tests/Fixtures/pins.json", {"a": 0.10000000000000003, "n": 3, "rows": [1.0000000000000002, -3e-16],
                                                "python": "3.12.3", "checks": ["ok  far field: 0.000e+00 <= 1e-13"],
                                                "name": "Si"})
        result = self.run_compare()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("float noise", result.stdout)

    def test_moved_number_fails(self):
        self.write("Tests/Fixtures/pins.json", {"a": 0.1000001, "n": 3, "rows": [1.0, 2.5e-17], "python": "3.12.14",
                                                "checks": ["ok  far field: 1.110e-16 <= 1e-13"], "name": "Si"})
        result = self.run_compare()
        self.assertEqual(result.returncode, 1)
        self.assertIn("DIFFERS", result.stdout)

    def test_integer_word_key_and_length_fail(self):
        for changed in ({"n": 4}, {"name": "Ge"}, {"extra": 1}, {"rows": [1.0]}, {"checks": ["FAIL far field: 1.110e-16 <= 1e-13"]}):
            value = {"a": 0.1, "n": 3, "rows": [1.0, 2.5e-17], "python": "3.12.14",
                     "checks": ["ok  far field: 1.110e-16 <= 1e-13"], "name": "Si", **changed}
            self.write("Tests/Fixtures/pins.json", value)
            self.assertEqual(self.run_compare().returncode, 1, changed)

    def test_new_and_deleted_files_fail(self):
        self.write("Tests/Fixtures/new.json", {"a": 1})
        self.assertEqual(self.run_compare().returncode, 1)
        os.remove(os.path.join(self.root, "Tests/Fixtures/new.json"))
        os.remove(os.path.join(self.root, "Tests/Fixtures/pins.json"))
        self.assertEqual(self.run_compare().returncode, 1)

    def test_text_file_numbers(self):
        self.write("Tests/Fixtures/table.swift", None, raw=b"let x = 0.1\nlet n = 3\n")
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "y"], cwd=self.root, check=True)
        self.write("Tests/Fixtures/table.swift", None, raw=b"let x = 0.10000000000000002\nlet n = 3\n")
        self.assertEqual(self.run_compare().returncode, 0)
        self.write("Tests/Fixtures/table.swift", None, raw=b"let x = 0.1\nlet n = 4\n")
        self.assertEqual(self.run_compare().returncode, 1)

    def test_binary_arrays(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is not installed")
        index = {"file": "a.bin", "sha256": "x", "index": {"v": {"offset": 0, "count": 2, "shape": [2], "kind": "f8"}}}
        self.write("Tests/Fixtures/a.json", {"arrays": index})
        self.write("Tests/Fixtures/a.bin", None, raw=struct.pack("<2d", 1.0, 1e-32))
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "y"], cwd=self.root, check=True)
        self.write("Tests/Fixtures/a.json", {"arrays": {**index, "sha256": "y"}})
        self.write("Tests/Fixtures/a.bin", None, raw=struct.pack("<2d", 1.0000000000000002, 3e-32))
        result = self.run_compare()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.write("Tests/Fixtures/a.bin", None, raw=struct.pack("<2d", 1.00001, 1e-32))
        self.assertEqual(self.run_compare().returncode, 1)


if __name__ == "__main__":
    unittest.main()
