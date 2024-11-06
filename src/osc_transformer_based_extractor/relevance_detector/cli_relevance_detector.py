import typer
import os
from .fine_tune import (
    check_csv_columns,
    check_output_dir,
    fine_tune_model,
)
from .inference import validate_path_exists, run_full_inference

# Subcommand app for relevance_detector
relevance_detector_app = typer.Typer()


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
    learning_rate: float = typer.Argument(..., help="Learning rate for training."),
    output_dir: str = typer.Argument(
        ..., help="Directory to save the fine-tuned model."
    ),
    export_model_name: str = typer.Argument(..., help="Name of the model to export."),
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
        learning_rate=learning_rate,
        output_dir=output_dir,
        export_model_name=export_model_name,
        save_steps=save_steps,
    )
    saved_model_path = os.path.join(output_dir, f"{export_model_name}")
    typer.echo(
        f"Model '{model_name}' is trained and saved successfully at {saved_model_path}"
    )


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
    batch_size: int = typer.Argument(16, help="Batch size to process the rows"),
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
            batch_size=batch_size,
            threshold=threshold,
        )

        typer.echo("Inference completed successfully!")

    except ValueError as ve:
        typer.echo(f"Error: {str(ve)}")
        raise typer.Exit(code=1)
