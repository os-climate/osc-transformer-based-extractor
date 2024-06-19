import argparse
import pandas as pd
import torch
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
import os


def check_csv_columns(file_path):
    """
    Check if the CSV file exists and contains the required columns.

    Args:
        file_path (str): Path to the CSV file.

    Raises:
        ValueError: If the file does not exist or does not contain the required columns.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"Data path {file_path} does not exist.")

    required_columns = ["question", "context", "label"]

    try:
        df = pd.read_csv(file_path)
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV file must contain the columns: {required_columns}. Missing columns: {missing_columns}")
    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty.")
    except pd.errors.ParserError:
        raise ValueError("Error parsing CSV file.")
    except Exception as e:
        raise ValueError(f"Error while checking CSV columns: {e}")


def check_output_dir(output_dir):
    """
    Check if the output directory is valid.

    Args:
        output_dir (str): Path to the output directory.

    Raises:
        ValueError: If the directory does not exist or is not a directory.
    """
    if not os.path.exists(output_dir):
        raise ValueError(f"Output directory {output_dir} does not exist.")
    if not os.path.isdir(output_dir):
        raise ValueError(f"Output path {output_dir} is not a directory.")


class CustomDataset(Dataset):
    def __init__(self, tokenizer, questions, contexts, labels, max_length):
        self.tokenizer = tokenizer
        self.questions = questions
        self.contexts = contexts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = str(self.questions[idx])
        context = str(self.contexts[idx])
        label = self.labels[idx]

        inputs = self.tokenizer(
            question, context, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )

        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
        }


def fine_tune_model(data_path, model_name, num_labels, max_length, epochs, batch_size, output_dir, save_steps):
    # Load your dataset into a pandas DataFrame
    df = pd.read_csv(data_path)
    df = df[["question", "context", "label"]]

    # Load Model and Tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Split the data into training and evaluation sets
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)

    # Create training and evaluation datasets
    train_dataset = CustomDataset(tokenizer, train_df["question"], train_df["context"], train_df["label"], max_length)
    eval_dataset = CustomDataset(tokenizer, eval_df["question"], eval_df["context"], eval_df["label"], max_length)

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=save_steps,
        logging_dir="./logs",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Start Training
    trainer.train()

    # Evaluate the model
    eval_result = trainer.evaluate(eval_dataset)
    print("Evaluation results:")
    for key, value in eval_result.items():
        print(f"{key}: {value}")

    # Predict labels for the evaluation dataset
    predictions = trainer.predict(eval_dataset)
    predicted_labels = predictions.predictions.argmax(axis=1)
    true_labels = [item["labels"].item() for item in eval_dataset]

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print("Accuracy:", accuracy)

    # Print inputs along with predicted labels
    for i, eva_data in enumerate(eval_dataset):
        input_ids = eva_data["input_ids"]
        true_label = true_labels[i]
        predicted_label = predicted_labels[i]
        print(f"Input: {tokenizer.decode(input_ids, skip_special_tokens=True)}")
        print(f"True Label: {true_label}, Predicted Label: {predicted_label}")
        print()

    # Save the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face model on a custom dataset.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV file containing the dataset.")
    parser.add_argument("--model_name", type=str, required=True, help="Name/ path of the pre-trained model to use(local or hugging face).")
    parser.add_argument("--num_labels", type=int, required=True, help="Number of labels for the classification task.")
    parser.add_argument("--max_length", type=int, required=True, help="Maximum length of the input sequences.")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the model will be saved during training.")
    parser.add_argument("--save_steps", type=int, required=True, help="Number of steps before saving the model during training.")

    args = parser.parse_args()

    check_csv_columns(args.data_path)
    check_output_dir(args.output_dir)

    fine_tune_model(
        data_path=args.data_path,
        model_name=args.model_name,
        num_labels=args.num_labels,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        save_steps=args.save_steps
    )

    print(f"Model- {args.model_name} Trained and Saved Successfully at {args.output_dir}")

'''
To run the file in CMD

python fine_tune.py \
  --data_path "data/train_data.csv" \
  --model_name "sentence-transformers/all-MiniLM-L6-v2" \
  --num_labels 2 \
  --max_length 512 \
  --epochs 2 \
  --batch_size 4 \
  --output_dir "./saved_models_during_training" \
  --save_steps 500

'''
