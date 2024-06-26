#############################################
OSC Transformer Based Extractor
#############################################

|osc-climate-project| |osc-climate-slack| |osc-climate-github| |pypi| |build-status| |pdm| |PyScaffold|



***********************************
OS-Climate Data Extraction Tool
***********************************


This project provides an CLI tool and python scripts to train a HuggingFace Transformer model or a local Transformer model and perform inference with it. The primary goal of the inference is to determine the relevance between a given question and context.

Installation
^^^^^^^^^^^^^

To install the OSC Transformer Based Extractor CLI, use pip:

.. code-block:: shell

    $ pip install osc-transformer-based-extractor

Alternatively, you can clone the repository from GitHub for a quick start:

.. code-block:: shell

    $ git clone https://github.com/os-climate/osc-transformer-based-extractor/


***************
Training Data
***************
To train the model, you need data from the curator module. The data is in CSV format. You can train the model either using the CLI or by calling the function directly in Python.
To train the model, you need data from the curator module. The data is in CSV format. You can train the model either using the CLI or by calling the function directly in Python.
Sample Data:

.. list-table:: Company Information
   :header-rows: 1

   * - Question
     - Context
     - Label
     - Company
     - Source File
     - Source Page
     - KPI ID
     - Year
     - Answer
     - Data Type
     - Annotator
     - Index
   * - What is the company name?
     - The Company is exposed to a risk of by losses counterparties their contractual financial obligations when due, and in particular depends on the reliability of banks the Company deposits its available cash.
     - 0
     - NOVATEK
     - 04_NOVATEK_AR_2016_ENG_11.pdf
     - ['0']
     - 0
     - 2016
     - PAO NOVATEK
     - TEXT
     - train_anno_large.xlsx
     - 1022




***************
CLI Usage
***************

The CLI command `osc-transformer-based-extractor` provides two main functions: training and inference. You can access detailed information about these functions by running the CLI command without any arguments.

**Commands**



* ``fine-tune``  :  Fine-tune a pre-trained Hugging Face model on a custom dataset.
* ``perform-inference`` :  Perform inference using a pre-trained sequence classification model.

* ``fine-tune``  :  Fine-tune a pre-trained Hugging Face model on a custom dataset.
* ``perform-inference`` :  Perform inference using a pre-trained sequence classification model.



************************
Using Github Repository
************************

Setting Up the Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To set up the working environment for this repository, follow these steps:

1. **Clone the repository**:

.. code-block:: shell

	$ git clone https://github.com/os-climate/osc-transformer-based-extractor/
    $ cd osc-transformer-based-extractor



2. **Create a new virtual environment and activate it**:

.. code-block:: shell

   		$ python -m venv venv
   		$ source venv/bin/activate  # On Windows use `venv\Scripts\activate`



3. **Install PDM**:

.. code-block:: shell

   		$ pip install pdm



4. **Sync the environment using PDM**:

.. code-block:: shell

   		$ pdm sync



5. **Add any new library**:

.. code-block:: shell

   		$ pdm add <library-name>


Train the model
^^^^^^^^^^^^^^^^^^^^^^^^^

To train the model, you can use the following code snippet:

.. code-block:: shell

    $ python fine_tune.py \
      --data_path "data/train_data.csv" \
      --model_name "sentence-transformers/all-MiniLM-L6-v2" \
      --num_labels 2 \
      --max_length 512 \
      --epochs 2 \
      --batch_size 4 \
      --output_dir "./saved_models_during_training" \
      --save_steps 500

OR use function calling:

.. code-block:: python

    from fine_tune import fine_tune_model


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

**Parameters**

* ``data_path (str)`` : Path to the training data CSV file.
* ``model_name (str)`` : Pre-trained model name from HuggingFace.
* ``num_labels (int)`` : Number of labels for the classification task.
* ``max_length (int)`` : Maximum sequence length.
* ``epochs (int)`` : Number of training epochs.
* ``batch_size (int)`` : Batch size for training.
* ``output_dir (str)`` : Directory to save the trained models.
* ``save_steps (int)`` : Number of steps between saving checkpoints.


Performing Inference
^^^^^^^^^^^^^^^^^^^^^^^^^

To perform inference and determine the relevance between a question and context, use the following code snippet:

.. code-block:: python

  $ python inference.py
      --question "What is the capital of France?"
      --context "Paris is the capital of France."
      --model_path /path/to/model
      --tokenizer_path /path/to/tokenizer

OR use function calling:

.. code-block:: python

  from inference import get_inference


  result = get_inference(
      question="What is the relevance?",
      context="This is a sample paragraph.",
      model_path="path/to/model",
      tokenizer_path="path/to/tokenizer" )


**Parameters**

* ``question (str)`` : The question for inference.
* ``context (str)`` : The paragraph to be analyzed.
* ``model_path (str)`` : Path to the pre-trained model.
* ``tokenizer_path (str)`` : Path to the tokenizer of the pre-trained model.



************************
Developer Notes
************************

For adding new dependencies use pdm. First install via pip::

    $ pip install pdm

And then you could add new packages via pdm add. For example numpy via::

    $ pdm add numpy

For running linting tools just to the following::

    $ pip install tox
    $ tox -e lint
    $ tox -e test



************************
Contributing
************************

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

All contributions (including pull requests) must agree to the Developer Certificate of Origin (DCO) version 1.1. This is exactly the same one created and used by the Linux kernel developers and posted on http://developercertificate.org/. This is a developer's certification that he or she has the right to submit the patch for inclusion into the project. Simply submitting a contribution implies this agreement, however, please include a "Signed-off-by" tag in every patch (this tag is a conventional way to confirm that you agree to the DCO).


On June 26 2024, Linux Foundation announced the merger of its financial services umbrella, the Fintech Open Source Foundation ([FINOS](https://finos.org)), with OS-Climate, an open source community dedicated to building data technologies, modeling, and analytic tools that will drive global capital flows into climate change mitigation and resilience; OS-Climate projects are in the process of transitioning to the [FINOS governance framework](https://community.finos.org/docs/governance); read more on [finos.org/press/finos-join-forces-os-open-source-climate-sustainability-esg](https://finos.org/press/finos-join-forces-os-open-source-climate-sustainability-esg)







.. |osc-climate-project| image:: https://img.shields.io/badge/OS-Climate-blue
  :alt: An OS-Climate Project
  :target: https://os-climate.org/

.. |osc-climate-slack| image:: https://img.shields.io/badge/slack-osclimate-brightgreen.svg?logo=slack
  :alt: Join OS-Climate on Slack
  :target: https://os-climate.slack.com

.. |osc-climate-github| image:: https://img.shields.io/badge/GitHub-100000?logo=github&logoColor=white
  :alt: Source code on GitHub
  :target: https://github.com/ModeSevenIndustrialSolutions/osc-data-extractor

.. |pypi| image:: https://img.shields.io/pypi/v/osc-data-extractor.svg
  :alt: PyPI package
  :target: https://pypi.org/project/osc-data-extractor/

.. |build-status| image:: https://api.cirrus-ci.com/github/os-climate/osc-data-extractor.svg?branch=main
  :alt: Built Status
  :target: https://cirrus-ci.com/github/os-climate/osc-data-extractor

.. |pdm| image:: https://img.shields.io/badge/PDM-Project-purple
  :alt: Built using PDM
  :target: https://pdm-project.org/latest/

.. |PyScaffold| image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
  :alt: Project generated with PyScaffold
  :target: https://pyscaffold.org/
