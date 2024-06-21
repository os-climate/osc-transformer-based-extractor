import os
import pandas as pd
import pytest
from osc_transformer_based_extractor.make_training_data_from_curator import (
    check_curator_data_path,
    check_kpi_mapping_path,
    check_output_path,
    make_training_data
)

# Define test data paths
curator_data_path_valid = r"src\PYTESTS\testing_data\valid_curator_data.csv"
curator_data_path_invalid = "invalid_curator_data.csv"
kpi_mapping_path_valid = r"src\PYTESTS\testing_data\valid_kpi_mapping.csv"
kpi_mapping_path_invalid = "invalid_kpi_mapping.csv"
output_path_valid = r"src\PYTESTS\testing_data\testt"
output_path_invalid = "invalid_output_path/file.txt"

# Ensure directory exists before saving CSV file
os.makedirs(os.path.dirname(curator_data_path_valid), exist_ok=True)
os.makedirs(output_path_valid, exist_ok=True)

# Create dummy CSV files for testing
dummy_curator_data = pd.DataFrame({"context": ["Lorem ipsum"], "kpi_id": [1], "label": [1]})
dummy_curator_data.to_csv(curator_data_path_valid, index=False)

dummy_kpi_mapping = pd.DataFrame({
    "kpi_id": [1],
    "question": ["What is the question?"],
    "sectors": ["Sector"],
    "add_year": [2022],
    "kpi_category": ["Category"]
})
dummy_kpi_mapping.to_csv(kpi_mapping_path_valid, index=False)


def test_check_curator_data_path():
    # Test valid path
    assert check_curator_data_path(curator_data_path_valid) is None

    # Test invalid path
    with pytest.raises(ValueError):
        check_curator_data_path(curator_data_path_invalid)


def test_check_kpi_mapping_path():
    # Test valid path
    assert check_kpi_mapping_path(kpi_mapping_path_valid) is None

    # Test invalid path
    with pytest.raises(ValueError):
        check_kpi_mapping_path(kpi_mapping_path_invalid)


def test_check_output_path():
    # Test valid path
    assert check_output_path(output_path_valid) is None

    # Test invalid path (file instead of directory)
    with pytest.raises(ValueError):
        check_output_path(output_path_invalid)

    # Test non-existent path
    non_existent_path = r"src\PYTESTS\testing_data\non_existent_directory"
    with pytest.raises(ValueError):
        check_output_path(non_existent_path)


def test_make_training_data():
    # Test generating training data
    make_training_data(curator_data_path_valid, kpi_mapping_path_valid, output_path_valid)
    output_file = os.path.join(output_path_valid, "train_data.csv")
    assert os.path.exists(output_file)

    # Validate contents of the generated file
    train_data = pd.read_csv(output_file)
    assert not train_data.empty
    assert 'question' in train_data.columns
    assert 'paragraph' in train_data.columns
    assert 'label' in train_data.columns

    # Clean up
    os.remove(output_file)


def test_invalid_kpi_id():
    # Create data with invalid kpi_id
    dummy_curator_data_invalid_kpi = pd.DataFrame({"context": ["Lorem ipsum"], "kpi_id": [999], "label": [1]})
    invalid_curator_data_path = r"src\PYTESTS\testing_data\invalid_curator_data.csv"
    dummy_curator_data_invalid_kpi.to_csv(invalid_curator_data_path, index=False)

    with pytest.raises(KeyError):
        make_training_data(invalid_curator_data_path, kpi_mapping_path_valid, output_path_valid)

    # Clean up
    os.remove(invalid_curator_data_path)


os.remove(os.path.dirname(curator_data_path_valid), exist_ok=True)
os.remove(output_path_valid, exist_ok=True)


if __name__ == "__main__":
    pytest.main()
