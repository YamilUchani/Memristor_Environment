from pathlib import Path

import pytest

from neurolab.validation.jo2010_ltp_ltd import get_default_path, load_data
from neurolab.validation.jo2010_stdp import (
    get_csv_path,
    load_jo2010_stdp_data,
)


def test_default_jo2010_reference_csvs_load():
    pulses, current, ltp_source = load_data()
    delta_t, delta_w, stdp_source = load_jo2010_stdp_data()

    assert get_default_path().is_file()
    assert get_csv_path().is_file()
    assert len(pulses) == len(current) == 87
    assert len(delta_t) == len(delta_w) == 13
    assert ltp_source == stdp_source == "csv"


@pytest.mark.parametrize(
    "loader",
    [load_data, load_jo2010_stdp_data],
)
def test_missing_jo2010_reference_csv_raises(loader, tmp_path: Path):
    missing_csv = tmp_path / "missing-reference.csv"

    with pytest.raises(FileNotFoundError, match="reference data not found"):
        loader(missing_csv)
