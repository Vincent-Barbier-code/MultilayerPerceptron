"""Tests for the data_extractor module."""

import unittest
import os
import pandas as pd

# Import the module to test
from data_processing import data_extractor as extractor


class TestDataExtractor(unittest.TestCase):
    """Tests for the data_extractor module."""

    def test_extract_data(self) -> None:
        """Test the extract_data function"""
        # Create a test CSV file
        test_csv = "test_data.csv"
        with open(test_csv, "w", encoding="utf-8") as file:
            file.write("col1,col2\n")
            file.write("1,2\n")
            file.write("3,4\n")

        # Call the extract_data function with the test file
        result = extractor.extract_data(test_csv)

        # Verify that the result is a pandas DataFrame
        self.assertIsInstance(result, pd.DataFrame)

        # Verify that the DataFrame has the correct columns
        expected_columns = [0, 1]
        self.assertListEqual(list(result.columns), expected_columns)

        # Verify that the DataFrame has the correct values
        expected_values = [["col1", "col2"], ["1", "2"], ["3", "4"]]
        self.assertListEqual(result.values.tolist(), expected_values)

        # Remove the test file
        os.remove(test_csv)

    def test_extract_data_empty_file(self) -> None:
        """Test the extract_data function with an empty file"""
        test_csv = "empty_test_data.csv"
        open(test_csv, "w", encoding="utf-8").close()

        result = extractor.extract_data(test_csv)
        self.assertIsInstance(result, pd.DataFrame)
        os.remove(test_csv)

    def test_extract_data_missing_file(self) -> None:
        """Test the extract_data function with a missing file"""
        result = extractor.extract_data("missing_file.csv")

        self.assertIsNone(result)
        