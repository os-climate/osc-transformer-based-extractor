import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.osc_transformer_based_extractor.fine_tune import check_csv_columns, check_output_dir, fine_tune_model


# Mock the Trainer class methods
class MockTrainer:
    def train(self):
        pass

    def evaluate(self, dataset):
        return {"eval_loss": 0.1, "eval_accuracy": 0.95}

@pytest.fixture
def mock_trainer():
    return MockTrainer()


# Test check_csv_columns function
def test_check_csv_columns_valid(tmp_path):
    # Create a temporary CSV file with required columns
    csv_path = tmp_path / "test_valid.csv"
    df = pd.DataFrame({"question": ["What is this?"], "paragraph": ["This is a paragraph."], "label": [1]})
    df.to_csv(csv_path, index=False)

    try:
        check_csv_columns(csv_path)
    except ValueError:
        pytest.fail("check_csv_columns raised ValueError unexpectedly.")


def test_check_csv_columns_missing_columns(tmp_path):
    # Create a temporary CSV file with missing columns
    csv_path = tmp_path / "test_missing.csv"
    df = pd.DataFrame({"question": ["What is this?"], "wrong_column": ["This is a paragraph."], "label": [1]})
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match=r".*Missing columns.*"):
        check_csv_columns(csv_path)


def test_check_output_dir_valid(tmp_path):
    # Existing directory
    output_dir = tmp_path / "output"
    os.makedirs(output_dir)

    try:
        check_output_dir(output_dir)
    except ValueError:
        pytest.fail("check_output_dir raised ValueError unexpectedly.")


def test_check_output_dir_invalid(tmp_path):
    # Non-existent directory
    output_dir = tmp_path / "non_existent_output"

    with pytest.raises(ValueError, match=r".*does not exist.*"):
        check_output_dir(output_dir)


# Test fine_tune_model function
@patch('relevance_detector.fine_tune.Trainer', autospec=True)
def test_fine_tune_model(mock_trainer, tmp_path):
    # Mocking Trainer class
    mock_trainer_instance = MagicMock()
    mock_trainer.return_value = mock_trainer_instance
    mock_trainer_instance.evaluate.return_value = {"eval_loss": 0.1, "eval_accuracy": 0.95}
    mock_trainer_instance.predict.return_value = MagicMock(predictions=MagicMock(argmax=lambda axis: [0, 1]),
                                                           label_ids=[0, 1])

    # Mocking model and tokenizer save methods
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_model.save_pretrained = MagicMock()
    mock_tokenizer.save_pretrained = MagicMock()

    # Test fine_tune_model
    fine_tune_model(
        data_path=str(tmp_path / "test_data.csv"),
        model_name="mock_model",
        num_labels=2,
        max_length=512,
        epochs=2,
        batch_size=4,
        output_dir=str(tmp_path / "saved_models"),
        save_steps=500
    )

    # Assertions for trainer methods
    mock_trainer_instance.train.assert_called_once()
    mock_trainer_instance.evaluate.assert_called_once()

    # Assertions for model and tokenizer save methods
    mock_model.save_pretrained.assert_called_once_with(str(tmp_path / "saved_models"))
    mock_tokenizer.save_pretrained.assert_called_once_with(str(tmp_path / "saved_models"))
