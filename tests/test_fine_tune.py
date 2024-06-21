from unittest.mock import patch, MagicMock
import pytest
import pandas as pd
import torch
import os
from transformers import Trainer, AutoModelForSequenceClassification, AutoTokenizer
from src.osc_transformer_based_extractor.fine_tune import check_csv_columns, check_output_dir, fine_tune_model

# Sample data for testing
data = {
    "question": ["What is AI?", "Explain machine learning."],
    "context": ["AI is the field of study focused on creating intelligent agents.", "Machine learning is a subset of AI focused on learning from data."],
    "label": [1, 0]
}
df = pd.DataFrame(data)
mock_csv_path = "mock_data.csv"
mock_output_dir = "mock_output_dir"

# Create a mock CSV file
df.to_csv(mock_csv_path, index=False)

# Create a mock output directory
if not os.path.exists(mock_output_dir):
    os.makedirs(mock_output_dir)


@pytest.fixture
def mock_data():
    return df


@pytest.fixture
def mock_args():
    class Args:
        data_path = mock_csv_path
        model_name = "bert-base-uncased"
        num_labels = 2
        max_length = 128
        epochs = 1
        batch_size = 2
        output_dir = mock_output_dir
        save_steps = 10
    return Args()


@patch("transformers.AutoModelForSequenceClassification.from_pretrained")
@patch("transformers.AutoTokenizer.from_pretrained")
@patch("transformers.Trainer")
def test_fine_tune_model(mock_trainer, mock_tokenizer, mock_model, mock_data, mock_args):
    # Mock the model and tokenizer
    model_instance = MagicMock(spec=AutoModelForSequenceClassification)
    model_instance.to = MagicMock(return_value=model_instance)

    # Set the attributes on the model class to mimic a PyTorch model
    mock_model_class = MagicMock(spec=AutoModelForSequenceClassification)
    mock_model_class.module = "torch"
    mock_model_class.name = "PreTrainedModel"
    mock_model_class.return_value = model_instance

    mock_model.return_value = mock_model_class.return_value
    mock_tokenizer.return_value = MagicMock(spec=AutoTokenizer)

    # Mock the Trainer's train, evaluate, and predict methods
    mock_trainer_instance = MagicMock(spec=Trainer)
    mock_trainer_instance.train.return_value = None
    mock_trainer_instance.evaluate.return_value = {"eval_loss": 0.5}
    mock_trainer_instance.predict.return_value = MagicMock(predictions=torch.tensor([[0.5, 0.5], [0.6, 0.4]]))
    mock_trainer.return_value = mock_trainer_instance

    # Run the fine_tune_model function with mock arguments
    fine_tune_model(
        data_path=mock_args.data_path,
        model_name=mock_args.model_name,
        num_labels=mock_args.num_labels,
        max_length=mock_args.max_length,
        epochs=mock_args.epochs,
        batch_size=mock_args.batch_size,
        output_dir=mock_args.output_dir,
        save_steps=mock_args.save_steps
    )

    # Assert that the model and tokenizer were loaded correctly
    mock_model.assert_called_once_with(mock_args.model_name, num_labels=mock_args.num_labels)
    mock_tokenizer.assert_called_once_with(mock_args.model_name)

    # Assert that the Trainer's train, evaluate, and predict methods were called
    mock_trainer_instance.train.assert_called_once()
    mock_trainer_instance.evaluate.assert_called_once()
    mock_trainer_instance.predict.assert_called_once()


def test_check_csv_columns():
    # Test with valid CSV file
    check_csv_columns(mock_csv_path)

    # Test with invalid CSV file path
    with pytest.raises(ValueError, match="Data path invalid_path.csv does not exist."):
        check_csv_columns("invalid_path.csv")

    # Test with missing columns
    df_invalid = df.drop(columns=["label"])
    invalid_csv_path = "invalid_columns.csv"
    df_invalid.to_csv(invalid_csv_path, index=False)
    with pytest.raises(ValueError, match="CSV file must contain the columns: \\['question', 'context', 'label'\\]. Missing columns: \\['label'\\]"):
        check_csv_columns(invalid_csv_path)
    os.remove(invalid_csv_path)


def test_check_output_dir():
    # Test with valid directory
    check_output_dir(mock_output_dir)

    # Test with invalid directory path
    with pytest.raises(ValueError, match="Output directory invalid_output_dir does not exist."):
        check_output_dir("invalid_output_dir")

    # Test with a file path instead of a directory
    temp_file_path = "temp_file.txt"
    with open(temp_file_path, "w") as f:
        f.write("temp")
    with pytest.raises(ValueError, match="Output path temp_file.txt is not a directory."):
        check_output_dir(temp_file_path)
    os.remove(temp_file_path)


# Clean up mock CSV file and directory after tests
@pytest.fixture(scope="module", autouse=True)
def cleanup(request):
    def remove_files():
        if os.path.exists(mock_csv_path):
            os.remove(mock_csv_path)
        if os.path.exists(mock_output_dir):
            os.rmdir(mock_output_dir)
    request.addfinalizer(remove_files)
