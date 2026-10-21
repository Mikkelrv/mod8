import contextlib
import io
import unittest
from unittest.mock import patch

import pandas as pd

import main


class StudentChurnTests(unittest.TestCase):
    def test_training_and_completion_probabilities(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(
            main, "train_test_split", wraps=main.train_test_split
        ) as split:
            main.main()
        original = pd.read_csv(main.DATA_FILE, sep=";")
        expected = original["Line"].map({"EUX": 0, "HF": 1, "HHX": 2, "HTX": 3, "STX": 4})
        pd.testing.assert_series_equal(split.call_args.args[0]["Line"], expected)
        self.assertIn("Accuracy:", output.getvalue())
        self.assertIn("Confusion matrix", output.getvalue())
        self.assertIn("Completed", output.getvalue())

    def test_unknown_outcome_is_rejected(self):
        data = pd.DataFrame([{column: 1 for column in main.FEATURES}])
        data[main.TARGET] = "Unknown"
        with patch.object(main.pd, "read_csv", return_value=data):
            with self.assertRaisesRegex(ValueError, "only Completed or Stopped"):
                main.main()

    def test_unknown_line_is_rejected(self):
        data = pd.DataFrame([{column: 1 for column in main.FEATURES}] * 2)
        data[main.TARGET] = ["Completed", "Stopped"]
        data["Line"] = ["HF", "Unknown"]
        with patch.object(main.pd, "read_csv", return_value=data):
            with self.assertRaisesRegex(ValueError, "Line must contain only"):
                main.main()


if __name__ == "__main__":
    unittest.main()
