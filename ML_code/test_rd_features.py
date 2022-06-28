"""A test of the `calculateRDKFeatures.py` script.

To run:

```
py.test test_rd_features.py
```

See `data/test/rd_features/` for inputs and expected outputs.
"""
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy
import pandas

from calculateRDKFeatures import calculate_rdk_features


def test_make_rdkit_feature_matrix():

    expected_features = pandas.read_csv(
        Path("../../data/test/rd_features/output.csv"), sep=",", index_col=0
    )

    input_path = Path("../../data/test/rd_features/input.csv")

    with NamedTemporaryFile() as tmp_file:

        output_path = tmp_file.name

        calculate_rdk_features(input_path, output_path)

        output_features = pandas.read_csv(output_path, sep=",", index_col=0)

    assert numpy.all(expected_features.columns == output_features.columns)

    assert numpy.all(
        # First two columns are name and SMILES
        expected_features.to_numpy()[:, :2] == output_features.to_numpy()[:, :2]
    )
    assert numpy.allclose(
        expected_features.to_numpy()[:, 2:].astype(float),
        output_features.to_numpy()[:, 2:].astype(float)
    )
