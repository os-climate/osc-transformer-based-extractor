#############################################
OSC Transformer Based Extractor
#############################################

|osc-climate-project| |osc-climate-slack| |osc-climate-github| |pypi| |build-status| |pdm| |PyScaffold|



***********************************
OS-Climate Data Extraction Tool
***********************************


This project provides an CLI tool and python scripts to train a HuggingFace Transformer model or a local Transformer model and perform inference with it. The primary goal of the inference is to determine the relevance between a given question and context.

Quick Start
^^^^^^^^^^^^^

To install the OSC Transformer Based Extractor CLI, use pip:

.. code-block:: shell

    $ pip install osc-transformer-based-extractor

Afterwards you can use the tooling as a CLI tool by simply typing:

.. code-block:: shell

We are using typer to have a nice CLI tool here. All details and help will be shown in the CLI
tool itself and are not described here in more detail.

**Example**: Assume the folder structure is like that:

.. code-block:: text

    project/
    │
    ├── kpi_mapping.csv
    ├── training_data.csv
    ├── data/
    │   └── (json files for inference command)
    ├── model/
    │   └── (model-related files go here)
    |── saved__model/
    |   └── (output files trained models)
    ├── output/
    │   └── (ouput files from inference command)


Then you can now simply run (after installation of osc-transformer-based-extractor)
the following command to fine-tune the model on the data:

.. code-block:: shell

  $ osc-transformer-based-extractor relevance-detector fine-tune \
    --data_path "project/training_data.csv" \
    --model_name "bert-base-uncased" \
    --num_labels 2 \
    --max_length 128 \
    --epochs 3 \
    --batch_size 16 \
    --output_dir "project/saved__model/" \
    --save_steps 500

Also, the following command can be run to perform inference:

.. code-block:: shell

  $ osc-transformer-based-extractor relevance-detector perform-inference \
    --folder_path "project/data/" \
    --kpi_mapping_path "project/kpi_mapping.csv" \
    --output_path "project/output/" \
    --model_path "project/model/" \
    --tokenizer_path "project/model/" \
    --threshold 0.5


***************
Training Data
***************

Training File
^^^^^^^^^^^^^^^

To train the model, you need a CSV file with columns:
     * ``Question``
     * ``Context``
     * ``Label``

Also additionally, the output of the https://github.com/os-climate/osc-transformer-presteps module can also be used. the output will look like following
Sample Data:

.. list-table:: traning_Data.csv
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


KPI Mapping File
^^^^^^^^^^^^^^^^^^^^^
The Inference command will need a kpi-mapping.csv file, which looks like:

.. list-table:: kpi_mapping.csv
   :header-rows: 1

   * - kpi_id
     - question
     - sectors
     - add_year
     - kpi_category
   * - 1
     - In which year was the annual report or the sustainability report published?
     - OG, CM, CU
     - FALSE
     - TEXT




************************
Developer Notes
************************

Use code directly without CLI via Github Repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First clone the repository to your local environment::

    $ git clone https://github.com/os-climate/osc-transformer-based-extractor/

We are using pdm to manage the packages and tox for a stable test framework.
Hence, first install pdm (possibly in a virtual environment) via::

    $ pip install pdm

Afterwards sync you system via::

    $ pdm sync

Now you have multiple demos on how to go on. See folder
[here](demo)

pdm
---

For adding new dependencies use pdm. You could add new packages via pdm add.
For example numpy via::

    $ pdm add numpy

For a very detailed description check the homepage of the pdm project:

https://pdm-project.org/en/latest/


tox
---

For running linting tools we use tox which you run outside of your virtual environment::

    $ pip install tox
    $ tox -e lint
    $ tox -e test

This will automatically apply some checks on your code and run the provided pytests. See
more details on tox on the homepage of the tox project:

https://tox.wiki/en/4.16.0/

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
