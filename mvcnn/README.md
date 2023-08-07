MVCNN
==============================

Multiview classification of chloroplast cells

Project Organization
------------

    ├── Makefile           <- Makefile with commands like `make evaluate` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── raw            <- The original data dump.
    │       ├── Set3_clean
    │       ├── Set3_noisy                  
    │       ├── Real_data
    │           ├── validation    
    │           ├── test        <- Real test data with samples having less than 3 views deleted.           
    │                                          
    │
    ├── lightning_logs	            <- Stores the tensorboard log files	
    │   ├──cnn                      <- Stores the single-view model, train and evaluation logs
    │   ├── mvcnnscoresum           <- Stores the score fusion model, train and evaluation logs
    │   └── mvcnnlatefusion         <- Stores the late fusion model, train and evaluation logs
    │ 
    ├── conda_requirements.yaml      <- yaml file to install the packages
    │
    ├── pip_requirements.txt         <- The requirements file if pip manager is used instead of conda yaml file.
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    │   
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to generate datasets by reading data from data folder in root directory.
    │   │   ├── __init__.py
    │   │   ├── chloroplast_dataset.py              <- Reads the image data from Data->raw folder 
    │   │   │                                          and creates single-view and multi-view datasets.
    │   │   ├── Chloroplastdatamodule_cnn.py        <- Apply preprocessing and create train, validation 
    │   │   │                                          and test dataloaders for single-view.
    │   │   ├── Chloroplastdatamodule_mvcnn.py      <- Apply preprocessing and create train, validation 
    │   │   │                                          and test dataloaders for multi-view.
    │   │   │    
    │   │   └── split_dataset.py     <- Utility to split a folder of images radomly into two folders.
    │   │
    │   ├── models                    <- Scripts to train models and then use trained models to make
    │   │   │                            predictions
    │   │   ├── cf_matrix.py          <- Creates a confusion matrix figure from confusion matrix.
    │   │   ├── evaluate_model.py     <- Evaluate the trained models with eval_config settings.
    │   │   ├── network.py            <- Creates single-view CNN and MVCNN network models in pytorch.
    │   │   ├── train_config.yaml     <- Training settings
    │   │   ├── cnn_lit_module.py     <- Manages train, validation, test loop and optimisers settings for CNN.
    │   │   ├── __init__.py           <- Makes src a Python module
    │   │   ├── plot_utilities.py     <- Contains functions to plot single-view, multi-view batch.
    │   │   ├── train_model.py        <- Train the trained models with train_config settings. 
    │   │   ├── eval_config.yaml      <- Evaluation settings
    │   │   ├── mvcnn_lit_module.py   <- Manages train, validation, test loop and optimisers settings for MVCNN.
    │   │   │
    │
    ├── sv_baseline	                    <- Stores the single-view baseline models. 	
    │   ├── sv_noisydataset_baseline    <- single-view baseline model trained on noisy dataset
    │   ├── sv_cleandataset_baseline    <- single-view baseline model trained on clean dataset
    │   
    ├── .gitignore	                    <- Git ignore for large log files and other un=important files
                                            not to be tracked by git.

    
--------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

> **_NOTE:_**  The parts of the code with  the comment "**# extra**" are additional features and can be ignored. 

Project Guide
--------------

1. Introduction:

    Welcome to Multi-view classification of chloroplast cells project! This guide will help you understand the folder structure, the purpose of each component, and how to run the code to train and evaluate models.

2. Folder Structure:

    The project is organised into different folders, as shown in the tree structure with the function for each file.


3. Getting Started:

    To use this project, follow these steps:
    
    For Linux
    ---------

    Step 1: 
    
    Option 1(Recommended way): This requires conda package manager. 
    Create the conda environment and install the required packages with the following command.

    `make create_environment`

    Activate the environment:

    `conda activate mvcnn`

    Option 2: Use any other tool for managing the virtual environment. Required packages can be installed using pip:
    
    `make pip_requirements`

    Step 2: Data Preparation: Place raw data in the data/raw/ directory, organised into subfolders for each dataset. Ensure that each dataset folder contains images and corresponding labels. The images for a multi-view dataset are grouped according to the name of the images in Lexicographic order. Hence, the following directory structure and naming shall be followed:


        root/diamond/diamond_00000_0.png
        root/diamond/diamond_00000_1.png
        root/diamond/diamond_00000_2.png

        root/gyroid/gyroid_00000_0.png
        root/gyroid/gyroid_00000_1.png
        root/gyroid/gyroid_00000_2.png

    The required pre-processing is implemented in datamodule files in src/data directory.

    Step 3:  Model Training: Model training is executed by train_model.py, which can be executed directly or simply using the following command in the root directory of the project.

    `make train`

    The train_model.py script will use the configurations defined in train_config.yaml train the model. The trained models and training tensorboard logs are stored in **lightning_logs** directory.

    Copy the single-view trained model files from lightning_logs to single-view baseline, to keep them seperate.


    Step 4: Visualising training curves and confusion matrix can be performed by following command

    `make tensorboard`

    Step 5: Model evaluation: Model evaluation is executed by evaluate_model.py, can be executed by

    `make evaluate`

    The evaluate_model.py script will use the configurations defined in evaluate_config.yaml to test the trained models. The path of the trained model and other settings have to be edited before executing the evaluation. The evaluation results are also stored in the same folder where training logs are stored.


    For Windows
    ---------
    
    Since Windows does not natively support Makefile, commands must be executed by typing the complete commands.

    Step 1: 
    
    Option 1(Recommended way): This requires conda package manager. 
    Create the conda environment and install the required packages with the following command. 

    `conda env create -n mvcnn  -f  conda_requirements.yaml`

    Activate the environment:

    `conda activate mvcnn`

    Option 2: Use any other tool for managing the virtual environment. Required packages can be installed using pip:
    
    `python -m pip install -U pip setuptools wheel`

	`python -m pip install -r pip_requirements.txt`

    Step 2: Data Preparation: Place raw data in the data/raw/ directory, organized into subfolders for each dataset. Ensure that each dataset folder contains images and corresponding labels. The images for a multi-view dataset are grouped according to the name of images in Lexicographic order. Hence, the following directory structure and naming shall be followed:

        root/diamond/diamond_00000_0.png
        root/diamond/diamond_00000_1.png
        root/diamond/diamond_00000_2.png

        root/gyroid/gyroid_00000_0.png
        root/gyroid/gyroid_00000_1.png
        root/gyroid/gyroid_00000_2.png

    The required pre-processing is implemented in datamodule files in src/data directory.

    Step 3:  Model Training: Model training is executed by train_model.py, which can be executed directly or simply using the following command in the root directory of the project.

    `python -m src.models.train_model`

    The train_model.py script will use the configurations defined in train_config.yaml train the model. 
    The trianed models and training tensorboard logs are stored in **lightning_logs** directory.

    Copy the single-view trained model files from lightning_logs to single-view baseline, to keep them seperate.

    Step 4: Visualising training curves and confusion matrix can be performed by following command

    `tensorboard --logdir ./lightning_logs/`

    Step 5: Model evaluation: Model evaluation is executed by evaluate_model.py, can be executed by

    `python -m src.models.evaluate_model`

    The evaluate_model.py script will use the configurations defined in evaluate_config.yaml to test the trained models. The path of the trained model and other settings have to be edited before executing the evaluation. The evaluation results are also stored in same folder where training logs are stored.






