from pathlib import Path
import yaml

from time import perf_counter
import torch
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers

from src.models.network import CnnNetwork, MvCnnNetwork, MvCnnScoreNetwork
from src.models.cnn_lit_module import CNNClassifier
from src.data.Chloroplastdatamodule_cnn import ChloroplastDataModule
from src.models.mvcnn_lit_module import MVCNNClassifier
from src.data.Chloroplastdatamodule_mvcnn import MVChloroplastDataModule
from src.models.plot_utilities import captum_grad_shap

# folder to load config file
CONFIG_PATH = Path(__file__).resolve().parents[0].joinpath("eval_config.yaml")


def cnn_evaluate(
    model_ckpt_path,
    log_path,
    data_dir,
    config,
    network_type,
    inference: bool = False,
    test: bool = False,
    visualize_attributions: bool = False,
):
    """Train a cnn network with the evaluate config settings."""

    # Time counter
    start = perf_counter()

    if network_type == "cnn":
        # instantiate the cnn network
        network_model = CnnNetwork()
        # instantiate the lightning module
        model = CNNClassifier.load_from_checkpoint(
            model_ckpt_path,
            model=network_model,
            plot_preds=config["plot_preds"],
            map_location=torch.device(config["accelerator"]),
        )
        # instantiate the datamodule
        dm = ChloroplastDataModule(
            data_dir=data_dir,
            num_views=3,
            test_real_data=config["test_real_data"],
        )

    elif network_type == "mvcnn":
        if config["fusion_strategy"] == "latefusion":
            network_model = MvCnnNetwork(
                map_location=torch.device(config["accelerator"])
            )
        elif (
            config["fusion_strategy"] == "scoremax"
            or config["fusion_strategy"] == "scoresum"
            or config["fusion_strategy"] == "scoremul"
        ):
            network_model = MvCnnScoreNetwork(
                fusion_strategy=config["fusion_strategy"],
                map_location=torch.device(config["accelerator"]),
            )

        # instantiate the lightning module
        model = MVCNNClassifier.load_from_checkpoint(
            model_ckpt_path,
            model=network_model,
            fusion_strategy=config["fusion_strategy"],
            plot_preds=config["plot_preds"],
            map_location=torch.device(config["accelerator"]),
        )
        # instantiate the datamodule
        dm = MVChloroplastDataModule(
            data_dir=data_dir, test_real_data=config["test_real_data"]
        )

    # instantiate the logger
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_path, name="evaluation_logs")

    # instantiate the trainer class
    trainer = pl.Trainer(logger=tb_logger)

    if inference:
        dm.setup("predict")

        predictions = trainer.predict(model, dm.predict_dataloader())
        print(predictions)

    if test:
        dm.setup("test")
        test_metrics = trainer.test(model, dm.test_dataloader())
        model.conf_mat()

        print(test_metrics)

    if visualize_attributions:
        dm.setup("test")
        data_itr = iter(dm.test_dataloader())
        batch = next(data_itr)
        while batch:
            captum_grad_shap(batch, model, network_type, config["fusion_strategy"])
            batch = next(data_itr)

    test_time = perf_counter() - start

    tb_logger.log_metrics({"test time": test_time})
    print(f"Test time: {test_time}")


def main():
    # go to the parent directory
    parent_dir = Path(__file__).resolve().parents[2]
    config = load_config(CONFIG_PATH)

    # 2. load the specific checkpoint file to the model
    if config["network_type"] == "mvcnn":
        saved_model_folder = config["network_type"] + config["fusion_strategy"]
    else:
        saved_model_folder = config["network_type"]
    model_ckpt_path = parent_dir.joinpath(
        "lightning_logs", saved_model_folder, *config["model_ckpt_path"]
    )
    print(model_ckpt_path)

    log_path = model_ckpt_path.parents[1]

    # check if the file exists or not
    print(Path.is_file(model_ckpt_path))
    print(Path.is_dir(model_ckpt_path))

    # load the data directory
    data_dir = parent_dir.joinpath("data", "raw", "Data", config["data_set_name"])

    cnn_evaluate(
        model_ckpt_path,
        log_path,
        data_dir,
        config,
        network_type=config["network_type"],
        inference=config["inference"],
        test=config["test"],
        visualize_attributions=config["visualize_attributions"],
    )


# Function to load yaml configuration file
def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config


if __name__ == "__main__":
    main()
