import sys
import os
import typer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../osc_transformer_based_extractor')))
from fine_tune import check_csv_columns, check_output_dir, fine_tune_model
from inference import check_model_and_tokenizer_path, check_question_context, get_inference

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

'''@app.command()
def make_training_data_from_curator(curator_data_path: str, kpi_mapping_path: str, output_path: str):
    """
    Make training data from curator data.

    Example:
      python main.py make_training_data_from_curator data.csv mappings.csv output/
    """
    check_curator_data_path(curator_data_path)
    check_kpi_mapping_path(kpi_mapping_path)
    check_output_path(output_path)

    make_training_data(
        curator_data_path=curator_data_path,
        kpi_mapping_path=kpi_mapping_path,
        output_path=output_path
    )

    typer.echo(f"Training data created at: {output_path}")'''


@app.command()
def fine_tune(data_path: str, model_name: str, num_labels: int, max_length: int, epochs: int, batch_size: int, output_dir: str, save_steps: int):
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
def perform_inference(question: str, context: str, model_path: str, tokenizer_path: str):
    """
    Perform inference using a pre-trained sequence classification model.

    Example:
      python main.py perform_inference "What is the main idea?" "This is the context." trained_model/ tokenizer/
    """

    print(type(question), type(context))

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
