# Relevance Detector

This folder contains a set of scripts and notebooks designed to process data, train a sentence transformer model, and perform inferences to detect the relevance of folder contents. Below is a detailed description of each file and folder included in this repository.

## How to Use This Repository

**Prepare Training Data**:

- One must have data from the curator module, which is used for training of the model. The data from the curator module is a CSV file as follows:

### Example Snippet

| question                  | context                                                                                                                                                                                                       | company | source_file                   | source_page | kpi_id | year | answer      | data_type | relevant_paragraphs                | annotator             | Index | label |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ----------------------------- | ----------- | ------ | ---- | ----------- | --------- | ---------------------------------- | --------------------- | ----- | ----- |
| What is the company name? | The Company is exposed to a risk of by losses counterparties their contractual financial obligations when due, and in particular depends on the reliability of banks the Company deposits its available cash. | NOVATEK | 04_NOVATEK_AR_2016_ENG_11.pdf | ['0']       | 0      | 2016 | PAO NOVATEK | TEXT      | ["PAO NOVATEK ANNUAL REPORT 2016"] | train_anno_large.xlsx | 1022  | 0     |

- If you have CSV data from the curator module, run `make_training_data_from_curator.py` to process and save it in the `Data` folder.
- Alternatively, you can use `make_sample_training_data.ipynb` to generate sample data from a sample CSV file.

**Train the Model**:

- Use `train_sentence_transformer.ipynb` or `train_sentence_transformer.py` to train a sentence transformer model with the processed data from the `Data` folder and save it locally. Follow the steps in the notebook or script to configure and start the training process.

- To train the model using function calling

  ```python
  from train_sentence_transformer import fine_tune_model
  fine_tune_model(
     data_path="data/train_data.csv",
     model_name="sentence-transformers/all-MiniLM-L6-v2",
     num_labels=2,
     max_length=512,
     epochs=2,
     batch_size=4,
     output_dir="./saved_models_during_training",
     save_steps=500
  )
  ```

  **Parameters**:

  - `data_path (str)`: Path to the training data CSV file.
  - `model_name (str)`: Pre-trained model name from HuggingFace.
  - `num_labels (int)`: Number of labels for the classification task.
  - `max_length (int)`: Maximum sequence length.
  - `epochs (int)`: Number of training epochs.
  - `batch_size (int)`: Batch size for training.
  - `output_dir (str)`: Directory to save the trained models.
  - `save_steps (int)`: Number of steps between saving checkpoints.

- To train the model from the command line, run `fine_tune.py` with the required arguments:

  ```bash
  python fine_tune.py \
    --data_path "data/train_data.csv" \
    --model_name "sentence-transformers/all-MiniLM-L6-v2" \
    --num_labels 2 \
    --max_length 512 \
    --epochs 2 \
    --batch_size 4 \
    --output_dir "./saved_models_during_training" \
    --save_steps 500
  ```

**Perform Inference**:

- Use `inference_demo.ipynb` to perform inferences with your trained model. Specify the model and tokenizer paths (either local or from HuggingFace) and run the notebook cells to see the results.
- For programmatic inference, you can use the function provided in `inference.py`:

  ```python
  from inference import get_inference
  result = get_inference(question="What is the relevance?", paragraph="This is a sample paragraph.", model_path="path/to/model", tokenizer_path="path/to/tokenizer")
  ```

## Repository Contents

### Python Scripts

1. **`inference.py`**

   - This script contains the function to make inferences using the trained model.
   - **Usage**: Import this script and use the provided function to predict the relevance of new data.
   - **Example**:

     ```python
     from inference import get_inference
     result = get_inference(question="What is the relevance?", paragraph="This is a sample paragraph.", model_path="path/to/model", tokenizer_path="path/to/tokenizer")
     ```

     **Parameters**:

     - `question (str)`: The question for inference.
     - `paragraph (str)`: The paragraph to be analyzed.
     - `model_path (str)`: Path to the pre-trained model.
     - `tokenizer_path (str)`: Path to the tokenizer of the pre-trained model.

2. **`make_training_data_from_curator.py`**

   - This script processes CSV data obtained from a module named `curator` to make it suitable for training the model.
   - **Usage**: Run this script to generate training data from the curator's output and save it in the `Data` folder.

3. **`train_sentence_transformer.py`**

   - This script defines a function to train a sentence transformer model, which can be called from other scripts or notebooks.
   - **Usage**: Import and call the `fine_tune_model` function to train your model.
   - **Example**:

     ```python
     from train_sentence_transformer import fine_tune_model
     fine_tune_model(
         data_path="data/train_data.csv",
         model_name="sentence-transformers/all-MiniLM-L6-v2",
         num_labels=2,
         max_length=512,
         epochs=2,
         batch_size=4,
         output_dir="./saved_models_during_training",
         save_steps=500
     )
     ```

     **Parameters**:

     - `data_path (str)`: Path to the training data CSV file.
     - `model_name (str)`: Pre-trained model name from HuggingFace.
     - `num_labels (int)`: Number of labels for the classification task.
     - `max_length (int)`: Maximum sequence length.
     - `epochs (int)`: Number of training epochs.
     - `batch_size (int)`: Batch size for training.
     - `output_dir (str)`: Directory to save the trained models.
     - `save_steps (int)`: Number of steps between saving checkpoints.

4. **`fine_tune.py`**

   - This script allows you to train a sentence transformer model from the command line.
   - **Usage**: Run this script from the command line with the necessary arguments.
   - **Example**:

     ```bash
     python fine_tune.py \
       --data_path "data/train_data.csv" \
       --model_name "sentence-transformers/all-MiniLM-L6-v2" \
       --num_labels 2 \
       --max_length 512 \
       --epochs 2 \
       --batch_size 4 \
       --output_dir "./saved_models_during_training" \
       --save_steps 500
     ```

### Jupyter Notebooks

1. **`inference_demo.ipynb`**

   - A notebook to demonstrate how to perform inferences using a custom model and tokenizer.
   - **Features**: Allows specifying model and tokenizer paths, which can be local paths or HuggingFace paths.
   - **Usage**: Open this notebook and follow the instructions to test inference with your own models.

2. **`make_sample_training_data.ipynb`**

   - This notebook was used to create sample training data from a sample CSV file.
   - **Usage**: Open and run this notebook to understand the process of creating sample data for training.

3. **`train_sentence_transformer.ipynb`**
   - A notebook to train a sentence transformer model and save the trained model locally.
   - **Usage**: Open and execute this notebook to train your model using the prepared data and save the trained model for inference.

### Data Folder

- **`Data/`**
  - This folder contains the processed training data obtained from the `curator` module. It serves as the input for training the sentence transformer model.

## Setting Up the Environment

To set up the working environment for this repository, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/folder-relevance-detector.git
   cd folder-relevance-detector
   ```

2. **Create a new virtual environment and activate it**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install PDM**:

   ```bash
   pip install pdm
   ```

4. **Sync the environment using PDM**:

   ```bash
   pdm sync
   ```

5. **Add any new library**:

   ```bash
   pdm add <library-name>
   ```

## Requirements

- Python 3.x
- Jupyter Notebook
- Required Python packages (install via `pdm` as described above)

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

---

For further details and documentation, please refer to the individual scripts and notebooks provided in this repository.
