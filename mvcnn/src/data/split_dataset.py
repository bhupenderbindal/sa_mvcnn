# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import shutil
import numpy as np
"""
split a train folder in train and val folder in defined ratio
"""


@click.command()
# @click.option("--input_filepath", type=click.Path(exists=True))
# @click.argument("output_filepath", type=click.Path())
@click.option("--dataset_name", type= str, default = "Set1_clean")
def main(dataset_name):
    np.random.seed(21)
    logger = logging.getLogger(__name__)
    logger.info("splitting data set from train data into train and val")
    project_dir = Path(__file__).resolve().parents[2]
    
    raw_data_dir = project_dir.joinpath("data", "raw", "Data")
    train_data_dir = raw_data_dir.joinpath(dataset_name, "train")
    train_split_dir = raw_data_dir.joinpath(dataset_name, "train_splitted")
    val_split_dir = raw_data_dir.joinpath(dataset_name, "val_splitted")


    for child in train_data_dir.iterdir():
        # creating a list of allfiles from the class directory in train directory
        allfilenames = list(child.rglob("*.png"))
        np.random.shuffle(allfilenames)

        trainfilenames, valfilenames = np.split(np.array(allfilenames),[int(len(allfilenames)*0.7)])
        print(f"total images of class {child.name}: {len(allfilenames)}")
        print(f"train images of class {child.name}: {len(trainfilenames)}")
        print(f"val images of class {child.name}: {len(valfilenames)}")

        train_class_dir = train_split_dir.joinpath(child.name)
        train_class_dir.mkdir(parents=True, exist_ok=False)

        val_class_dir = val_split_dir.joinpath(child.name)
        val_class_dir.mkdir(parents=True, exist_ok=False)

        ## Copy pasting images to target directory

        for name in trainfilenames:
            shutil.copy(name, train_class_dir )


        for name in valfilenames:
            shutil.copy(name, val_class_dir )

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)  

    main()
