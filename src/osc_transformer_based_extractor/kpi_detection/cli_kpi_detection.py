import typer
import os
from .train_kpi_detection import (
    train_kpi_detection,
    check_output_dir,
    check_csv_columns_kpi_detection,
)
from .inference_kpi_detection import (
    run_full_inference_kpi_detection,
    validate_path_exists,
)

# Subcommand app for kpi_detection
kpi_detection_app = typer.Typer()


@kpi_detection_app.callback(invoke_without_command=True)
def kpi_detection(ctx: typer.Context):
    """
    Commands for kpi detection tasks.

    Available commands:
    - fine-tune
    - inference
    """
    if ctx.invoked_subcommand is None:
        typer.echo(kpi_detection.__doc__)
        raise typer.Exit()


@kpi_detection_app.command("fine-tune")
def fine_tune_qna(
    data_path: str = typer.Argument(
        ..., help="Path to the CSV file containing training data."
    ),
    model_name: str = typer.Argument(
        ...,
        help="Name of the pre-trained model on HuggingFace OR path to local model directory.",
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
    check_csv_columns_kpi_detection(data_path)
    check_output_dir(output_dir)

    train_kpi_detection(
        data_path=data_path,
        model_name=model_name,
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


@kpi_detection_app.command("inference")
def inference_qna(
    data_file_path: str = typer.Argument(
        ..., help="Path to the input CSV file containing the dataset."
    ),
    output_path: str = typer.Argument(
        ..., help="Path to the directory where the output Excel file will be saved."
    ),
    model_path: str = typer.Argument(
        ..., help="Path to the pre-trained model directory OR name on huggingface."
    ),
    batch_size: int = typer.Argument(16, help="The batch size for inference."),
):
    """Perform inference using a pre-trained model on a dataset of kpis and contexts, saving an output Excel file."""
    try:
        validate_path_exists(data_file_path, "data_file_path")
        validate_path_exists(output_path, "output_path")

        run_full_inference_kpi_detection(
            data_file_path=data_file_path,
            output_path=output_path,
            model_path=model_path,
            batch_size=batch_size,
        )

        typer.echo("Inference completed successfully!")

    except ValueError as ve:
        typer.echo(f"Error: {str(ve)}")
        raise typer.Exit(code=1)
