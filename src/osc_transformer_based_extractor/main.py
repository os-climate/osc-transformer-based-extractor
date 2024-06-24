"""
osc_transformer_based_extractor: CLI for transformer-based model tasks.

This module provides command-line interface (CLI) commands for performing tasks such
as fine-tuning a transformer model and performing inference using a pre-trained model.
The CLI is built using Typer, a library for creating command-line interfaces.

Example usage:
  python main.py fine_tune data.csv bert-base-uncased 5 128 3 32 trained_models/ 500
  python main.py perform_inference "What is the main idea?" "This is the context." trained_model/ tokenizer/
"""
import typer
from .fine_tune import check_csv_columns, check_output_dir, fine_tune_model
from .inference import check_model_and_tokenizer_path, check_question_context, get_inference

app = typer.Typer()


@app.callback(invoke_without_command=True)
def callback(ctx: typer.Context):
    """
    osc_transformer_based_extractor: CLI for transformer-based model tasks.

    Example usage:
      python main.py fine_tune data.csv bert-base-uncased 5 128 3 32 trained_models/ 500
      python main.py perform_inference "What is the main idea?" "This is the context." trained_model/ tokenizer/
    """
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


@app.command()
def fine_tune(
    data_path: str = typer.Argument(..., help="Path to the CSV file containing training data."),
    model_name: str = typer.Argument(..., help="Name of the pre-trained Hugging Face model."),
    num_labels: int = typer.Argument(..., help="Number of labels for the classification task."),
    max_length: int = typer.Argument(..., help="Maximum length of the sequences."),
    epochs: int = typer.Argument(..., help="Number of training epochs."),
    batch_size: int = typer.Argument(..., help="Batch size for training."),
    output_dir: str = typer.Argument(..., help="Directory to save the fine-tuned model."),
    save_steps: int = typer.Argument(..., help="Number of steps between saving model checkpoints.")
):
    """
    Fine-tune a pre-trained Hugging Face model on a custom dataset.

    Example:
      python main.py fine_tune data.csv bert-base-uncased 5 128 3 32 trained_models/ 500
    """
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
        save_steps=save_steps
    )

    typer.echo(f"Model '{model_name}' trained and saved successfully at {output_dir}")


@app.command()
def perform_inference(
    question: str = typer.Argument(..., help="The question to be answered."),
    context: str = typer.Argument(..., help="The context or paragraph related to the question."),
    model_path: str = typer.Argument(..., help="Path to the pre-trained model directory."),
    tokenizer_path: str = typer.Argument(..., help="Path to the tokenizer directory.")
):
    """
    Perform inference using a pre-trained sequence classification model.

    Example:
      python main.py perform_inference "What is the main idea?" "This is the context." trained_model/ tokenizer/
    """
    try:
        check_question_context(question, context)
        check_model_and_tokenizer_path(model_path, tokenizer_path)

        result = get_inference(
            question=question,
            context=context,
            model_path=model_path,
            tokenizer_path=tokenizer_path
        )

        typer.echo(f"Predicted Label ID: {result}")

    except ValueError as ve:
        typer.echo(f"Error: {str(ve)}")
        raise typer.Exit(code=1)  # Exit with a non-zero code to indicate error
