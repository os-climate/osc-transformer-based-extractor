import os
import pytest
import torch
from unittest.mock import patch, MagicMock
from src.osc_transformer_based_extractor.inference import check_model_and_tokenizer_path, get_inference, check_question_context

# Define test data paths
model_path_valid = "valid_model"
model_path_invalid = "invalid_model"
tokenizer_path_valid = "valid_tokenizer"
tokenizer_path_invalid = "invalid_tokenizer"

# Create dummy model and tokenizer directories for testing
os.makedirs(model_path_valid, exist_ok=True)
os.makedirs(tokenizer_path_valid, exist_ok=True)


def test_check_model_and_tokenizer_path():
    # Test valid paths
    assert check_model_and_tokenizer_path(model_path_valid, tokenizer_path_valid) is None

    # Test invalid model path
    with pytest.raises(ValueError):
        check_model_and_tokenizer_path(model_path_invalid, tokenizer_path_valid)

    # Test invalid tokenizer path
    with pytest.raises(ValueError):
        check_model_and_tokenizer_path(model_path_valid, tokenizer_path_invalid)


def test_check_question_context():
    # Test valid inputs
    check_question_context("What is the capital of France?", "Paris is the capital of France.")

    # Test invalid question type
    with pytest.raises(ValueError, match="Question must be a string."):
        check_question_context(123, "Paris is the capital of France.")

    # Test invalid context type
    with pytest.raises(ValueError, match="context must be a string."):
        check_question_context("What is the capital of France?", 123)

    # Test empty question
    with pytest.raises(ValueError, match="Question is Emtpy"):
        check_question_context("", "Paris is the capital of France.")

    # Test empty context
    with pytest.raises(ValueError, match="context is Emtpy"):
        check_question_context("What is the capital of France?", "")


@patch("relevance_detector.inference.AutoModelForSequenceClassification.from_pretrained")
@patch("relevance_detector.inference.AutoTokenizer.from_pretrained")
def test_get_inference(mock_tokenizer, mock_model):
    # Mock tokenizer and model
    tokenizer_mock = MagicMock()
    model_mock = MagicMock()
    mock_tokenizer.return_value = tokenizer_mock
    mock_model.return_value = model_mock

    # Configure the tokenizer mock
    tokenizer_mock.encode_plus.return_value = {
        "input_ids": torch.tensor([[101, 102]]),
        "attention_mask": torch.tensor([[1, 1]])
    }

    # Configure the model mock to return a tensor for logits
    model_output_mock = MagicMock()
    model_output_mock.logits = torch.tensor([[0.1, 0.9]])
    model_mock.return_value = model_output_mock

    # Dummy question and context
    question = "What is the capital of France?"
    context = "Paris is the capital of France."

    # Dummy model and tokenizer paths
    model_path = model_path_valid
    tokenizer_path = tokenizer_path_valid

    # Test inference
    predicted_label_id = get_inference(question, context, model_path, tokenizer_path)

    # Assert that predicted_label_id is an integer
    assert isinstance(predicted_label_id, int)

    # Test different inputs
    tokenizer_mock.encode_plus.return_value = {
        "input_ids": torch.tensor([[101, 103]]),
        "attention_mask": torch.tensor([[1, 1]])
    }
    model_output_mock.logits = torch.tensor([[0.7, 0.3]])
    predicted_label_id = get_inference("What is the capital of Germany?", "Berlin is the capital of Germany.", model_path, tokenizer_path)
    assert isinstance(predicted_label_id, int)


if __name__ == "__main__":
    pytest.main()
