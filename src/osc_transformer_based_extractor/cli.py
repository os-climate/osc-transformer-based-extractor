"""
osc-transformer-based-extractor: CLI for transformer-based model tasks.

This module provides command-line interface (CLI) commands for performing tasks such
as fine-tuning a transformer model and performing inference using a pre-trained model.
The CLI is built using Typer, a library for creating command-line interfaces.

Example usage:
  osc-transformer-based-extractor relevance-detector fine-tune data.csv bert-base-uncased 5 128 3 32 trained_models/ 500
  osc-transformer-based-extractor relevance-detector inference "/all_json/" "data/kpi_mapping.csv" "output_dir/"  "model/" "tokenizer/" 0.8
"""

import typer
from .relevance_detector.fine_tune import (
    check_csv_columns,
    check_output_dir,
    fine_tune_model,
)
from .relevance_detector.inference import validate_path_exists, run_full_inference

# Main Typer app
app = typer.Typer(name="osc-transformer-based-extractor")


# Define the callback for the main app to provide help information
@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """
    OSC Transformer Based Extractor CLI.

    Available commands:
    - relevance-detector
    """
    if ctx.invoked_subcommand is None:
        typer.echo(main.__doc__)
        raise typer.Exit()


# Subcommand app for relevance_detector
relevance_detector_app = typer.Typer()


# Define the callback for relevance_detector to provide help information
@relevance_detector_app.callback(invoke_without_command=True)
def relevance_detector(ctx: typer.Context):
    """
    Commands for relevance detector tasks.

    Available commands:
    - fine-tune
    - inference
    """
    if ctx.invoked_subcommand is None:
        typer.echo(relevance_detector.__doc__)
        raise typer.Exit()


# Command for fine-tuning the model
@relevance_detector_app.command("fine-tune")
def fine_tune(
    data_path: str = typer.Argument(
        ..., help="Path to the CSV file containing training data."
    ),
    model_name: str = typer.Argument(
        ...,
        help="Name of the pre-trained model on HuggingFace OR path to local model directory.",
    ),
    num_labels: int = typer.Argument(
        ..., help="Number of labels for the classification task."
    ),
    max_length: int = typer.Argument(..., help="Maximum length of the sequences."),
    epochs: int = typer.Argument(..., help="Number of training epochs."),
    batch_size: int = typer.Argument(..., help="Batch size for training."),
    output_dir: str = typer.Argument(
        ..., help="Directory to save the fine-tuned model."
    ),
    save_steps: int = typer.Argument(
        ..., help="Number of steps between saving model checkpoints."
    ),
):
    """Fine-tune a pre-trained Hugging Face model on a custom dataset."""
    check_csv_columns(data_path)
    check_output_dir(output_dir)

    fine_tune_model(
        data_path=data_path,
        model_name=model_name,
        num_labels=num_labels,
        max_length=max_length,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir,
        save_steps=save_steps,
    )

    typer.echo(f"Model '{model_name}' trained and saved successfully at {output_dir}")


# Command for performing inference
@relevance_detector_app.command("inference")
def inference(
    json_folder_path: str = typer.Argument(
        ..., help="Folder path where all JSONs are stored."
    ),
    kpi_mapping_path: str = typer.Argument(..., help="Path to the kpi_mapping.csv"),
    output_path: str = typer.Argument(
        ..., help="Path to the folder where the output Excel files will be saved."
    ),
    model_path: str = typer.Argument(
        ..., help="Path to the pre-trained model directory OR name on huggingface."
    ),
    tokenizer_path: str = typer.Argument(
        ..., help="Path to the tokenizer directory OR name on huggingface."
    ),
    threshold: float = typer.Argument(
        0.5, help="Threshold value for prediction confidence."
    ),
):
    """Perform inference using a pre-trained model on all JSON files and KPI mapping file, saving an output Excel file for each JSON file in a specified folder."""
    try:
        validate_path_exists(json_folder_path, "folder_path")
        validate_path_exists(kpi_mapping_path, "kpi_mapping_path")
        validate_path_exists(output_path, "output_path")

        run_full_inference(
            json_folder_path=json_folder_path,
            kpi_mapping_path=kpi_mapping_path,
            output_path=output_path,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            threshold=threshold,
        )

        typer.echo("Inference completed successfully!")

    except ValueError as ve:
        typer.echo(f"Error: {str(ve)}")
        raise typer.Exit(code=1)


# Add the relevance_detector app as a command group under the main app
app.add_typer(relevance_detector_app, name="relevance-detector")
