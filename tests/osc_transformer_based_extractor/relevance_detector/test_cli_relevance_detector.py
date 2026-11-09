from unittest.mock import patch

from typer.testing import CliRunner

from osc_transformer_based_extractor.relevance_detector.cli_relevance_detector import (
    relevance_detector_app,
)

runner = CliRunner()


def test_relevance_detector_no_command():
    result = runner.invoke(relevance_detector_app)
    assert result.exit_code == 0
    assert "Commands for relevance detector tasks." in result.output
    assert "- fine-tune" in result.output
    assert "- inference" in result.output


@patch("relevance_detector.check_csv_columns")
@patch("relevance_detector.check_output_dir")
@patch("relevance_detector.fine_tune_model")
def test_fine_tune_command(
    mock_fine_tune_model, mock_check_output_dir, mock_check_csv_columns, tmpdir
):
    data_path = tmpdir.join("data.csv")
    output_dir = tmpdir.join("output")
    data_path.write("sample data")
    output_dir.mkdir()

    result = runner.invoke(
        relevance_detector_app,
        [
            "fine-tune",
            str(data_path),
            "bert-base-uncased",
            "2",
            "128",
            "3",
            "16",
            str(output_dir),
            "500",
        ],
    )
    assert result.exit_code == 0
    assert (
        f"Model 'bert-base-uncased' trained and saved successfully at {output_dir}"
        in result.output
    )

    mock_check_csv_columns.assert_called_once_with(str(data_path))
    mock_check_output_dir.assert_called_once_with(str(output_dir))
    mock_fine_tune_model.assert_called_once_with(
        data_path=str(data_path),
        model_name="bert-base-uncased",
        num_labels=2,
        max_length=128,
        epochs=3,
        batch_size=16,
        output_dir=str(output_dir),
        save_steps=500,
    )


@patch("relevance_detector.validate_path_exists")
@patch("relevance_detector.run_full_inference")
def test_inference_command(mock_run_full_inference, mock_validate_path_exists, tmpdir):
    json_folder_path = tmpdir.mkdir("json_folder")
    kpi_mapping_path = tmpdir.join("kpi_mapping.csv")
    output_path = tmpdir.mkdir("output_folder")
    kpi_mapping_path.write("sample kpi data")

    result = runner.invoke(
        relevance_detector_app,
        [
            "inference",
            str(json_folder_path),
            str(kpi_mapping_path),
            str(output_path),
            "path/to/model",
            "path/to/tokenizer",
            "0.75",
        ],
    )
    assert result.exit_code == 0
    assert "Inference completed successfully!" in result.output

    mock_validate_path_exists.assert_any_call(str(json_folder_path), "folder_path")
    mock_validate_path_exists.assert_any_call(str(kpi_mapping_path), "kpi_mapping_path")
    mock_validate_path_exists.assert_any_call(str(output_path), "output_path")
    mock_run_full_inference.assert_called_once_with(
        json_folder_path=str(json_folder_path),
        kpi_mapping_path=str(kpi_mapping_path),
        output_path=str(output_path),
        model_path="path/to/model",
        tokenizer_path="path/to/tokenizer",
        threshold=0.75,
    )


@patch(
    "relevance_detector.validate_path_exists", side_effect=ValueError("Invalid path")
)
def test_inference_command_invalid_path(mock_validate_path_exists):
    result = runner.invoke(
        relevance_detector_app,
        [
            "inference",
            "invalid/json_folder",
            "path/to/kpi_mapping.csv",
            "path/to/output_folder",
            "path/to/model",
            "path/to/tokenizer",
            "0.75",
        ],
    )
    assert result.exit_code == 1
    assert "Error: Invalid path" in result.output
    mock_validate_path_exists.assert_called_once_with(
        "invalid/json_folder", "folder_path"
    )
