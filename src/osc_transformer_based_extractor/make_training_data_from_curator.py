"""Generate training data based on curator data and KPI mapping."""

import os
import argparse
import pandas as pd


def check_curator_data_path(data_path):
    """
    Check if curator data path exists and is a CSV file.

    Args:
        data_path (str): Path to the curator data file.

    Raises:
        ValueError: If the path does not exist or is not a CSV file.
    """
    if not os.path.exists(data_path):
        raise ValueError("Curator data path does not exist.")
    if not data_path.lower().endswith('.csv'):
        raise ValueError("Curator data path is not a CSV file.")


def check_kpi_mapping_path(data_path):
    """
    Check if KPI mapping path exists and is a CSV file.

    Args:
        data_path (str): Path to the KPI mapping file.

    Raises:
        ValueError: If the path does not exist or is not a CSV file.
    """
    if not os.path.exists(data_path):
        raise ValueError("KPI mapping path does not exist.")
    if not data_path.lower().endswith('.csv'):
        raise ValueError("KPI mapping path is not a CSV file.")


def check_output_path(output_path):
    """
    Check if the output path exists, if not create it.

    Args:
        output_path (str): Path to the output directory.

    Raises:
        ValueError: If the path is not a directory.
    """
    if not os.path.exists(output_path):
        raise ValueError("Output path does not exist.")

    if not os.path.isdir(output_path):
        raise ValueError("Output path is not a directory.")


def make_training_data(curator_data_path: str, kpi_mapping_path: str, output_path: str) -> None:
    """
    Generate training data based on curator data and KPI mapping.

    Args:
        curator_data_path (str): Path to the curator data CSV file.
        kpi_mapping_path (str): Path to the KPI mapping CSV file.
        output_path (str): Path to the output directory.

    Returns:
        None
    """
    data = pd.read_csv(curator_data_path)

    kpi_mapping_df = pd.read_csv(kpi_mapping_path)
    kpi_mapping_df = kpi_mapping_df.iloc[:, :5]

    kpi_dict = {}
    for _, row in kpi_mapping_df.iterrows():
        kpi_id = row["kpi_id"]
        if isinstance(kpi_id, float) and kpi_id.is_integer():
            kpi_id = int(kpi_id)
        kpi_dict[kpi_id] = {
            "question": row["question"],
            "sectors": row["sectors"],
            "add_year": row["add_year"],
            "kpi_category": row["kpi_category"],
        }

    # Initialize lists to store data for new DataFrame
    question_list = []
    paragraph_list = []
    label_list = []

    for i in range(data.shape[0]):
        paragraph = data["context"][i]
        kpi_id = data["kpi_id"][i]

        label = data["label"][i]
        question = kpi_dict[kpi_id]["question"]

        # Append values to lists
        question_list.append(question)
        paragraph_list.append(paragraph)
        label_list.append(label)

    # Create new DataFrame
    train_data = pd.DataFrame(
        {
            "question": question_list,
            "paragraph": paragraph_list,
            "label": label_list,
        }
    )

    # Save the DataFrame to the specified output path
    file_name = "train_data.csv"
    save_dir = os.path.join(output_path, file_name)
    train_data.to_csv(save_dir, index=False)

    print(f"Data saved at {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make Training Data for training the model using the output from the Curator module.")
    parser.add_argument("--curator_data_path", type=str, required=True, help="Path to the CSV file from the Curator Module.")
    parser.add_argument("--kpi_mapping_path", type=str, required=True, help="Path to the kpi_mapping CSV file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory.")

    args = parser.parse_args()

    check_curator_data_path(args.curator_data_path)
    check_kpi_mapping_path(args.kpi_mapping_path)
    check_output_path(args.output_path)

    make_training_data(
        curator_data_path=args.curator_data_path,
        kpi_mapping_path=args.kpi_mapping_path,
        output_path=args.output_path
    )

    print("Training Data Successfully Made !!")
