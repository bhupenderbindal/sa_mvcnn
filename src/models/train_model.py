import yaml
from pathlib import Path
import numpy as np
import random
import os
import lightning.pytorch as pl
from argparse import ArgumentParser
import torch

from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import Timer, EarlyStopping, ModelCheckpoint
from lightning.pytorch.callbacks import Callback

from lightning.pytorch.utilities.seed import isolate_rng
import optuna
from time import perf_counter

from src.models.network import MvCnnNetwork, MvCnnScoreNetwork
from src.data.Chloroplastdatamodule_mvcnn import MVChloroplastDataModule

from src.data.Chloroplastdatamodule_cnn import ChloroplastDataModule
from src.models.network import CnnNetwork
from src.models.cnn_lit_module import CNNClassifier
from src.models.mvcnn_lit_module import MVCNNClassifier


# folder to load config file
CONFIG_PATH = Path(__file__).resolve().parents[0].joinpath("train_config.yaml")


def cnn_train(parent_dir, data_dir, config, network_type, data_set, params):
    """Train a cnn network with the train config settings."""

    # Time counter
    start = perf_counter()
    if network_type == "cnn":
        # instantiate the cnn network
        network_model = CnnNetwork()
        # instantiate the lightning module
        model = CNNClassifier(model=network_model, lr=params["lr_optuna"])
        # instantiate the datamodule
        dm = ChloroplastDataModule(
            data_dir=data_dir,
            num_views=config["num_views"],
            rand_rot_angle=config["rand_rot_angle"],
        )
        # set the lighting logging directory
        lightning_logs_dir = parent_dir.joinpath("lightning_logs", "cnn")

        # instantiate the logger
        tb_logger = pl_loggers.TensorBoardLogger(
            name=data_set,
            save_dir=lightning_logs_dir,
            sub_dir="training_logs",
            default_hp_metric=False,
        )
        # early stopping can be implemented only for single mteric but sv has different val_loss metric
        early_stopping1 = EarlyStopping("val_loss/dataloader_idx_0", patience=10)
        early_stopping2 = EarlyStopping("val_loss/dataloader_idx_1", patience=10)
        early_stopping3 = EarlyStopping("val_loss/dataloader_idx_2", patience=10)
        early_stop_callback = [early_stopping1, early_stopping2, early_stopping3]

    elif network_type == "mvcnn":
        # instantiate the mvcnn network
        if config["fusion_strategy"] == "latefusion":
            network_model = MvCnnNetwork(
                num_views=config["num_views"], map_location=config["accelerator"]
            )
        elif (
            config["fusion_strategy"] == "scoremax"
            or config["fusion_strategy"] == "scoresum"
            or config["fusion_strategy"] == "scoremul"
        ):
            network_model = MvCnnScoreNetwork(
                num_views=config["num_views"],
                fusion_strategy=config["fusion_strategy"],
                map_location=config["accelerator"],
            )

        # instantiate the lightning module
        model = MVCNNClassifier(
            model=network_model,
            lr=params["lr_optuna"],
            fusion_strategy=config["fusion_strategy"],
        )

        # instantiate the datamodule
        dm = MVChloroplastDataModule(
            data_dir=data_dir,
            num_views=config["num_views"],
            rand_rot_angle=config["rand_rot_angle"],
        )
        # set the lighting logging directory
        lightning_logs_dir = parent_dir.joinpath(
            "lightning_logs", "mvcnn" + config["fusion_strategy"]
        )

        # instantiate the logger
        tb_logger = pl_loggers.TensorBoardLogger(
            name=data_set,
            save_dir=lightning_logs_dir,
            sub_dir="training_logs",
            default_hp_metric=False,
        )

        early_stop_callback = [EarlyStopping("val_loss", patience=10)]
    # provide a time dictionary with the maximum time
    timer = Timer(duration=dict(weeks=1, days=1))

    checkpoint_callback = ModelCheckpoint(monitor="train_loss", save_top_k=7)

    # instantiate the trainer class
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        log_every_n_steps=50,  # check time effect
        accelerator=config["accelerator"],
        devices=config["devices"],
        default_root_dir=lightning_logs_dir,
        logger=tb_logger,
        callbacks=[
            timer,
            checkpoint_callback,
            *early_stop_callback,
        ],
        num_sanity_val_steps=0,
    )

    trainer.fit(model, dm)
    total_time = perf_counter() - start
    training_time = timer.time_elapsed("train")

    tb_logger.log_metrics({"total time": total_time, "training time": training_time})
    print(f"Total time: {total_time}")
    # query training/validation/test time (in seconds)
    print(f"Training time: {training_time}")

    if network_type == "cnn":
        val_acc = [
            trainer.callback_metrics["val_acc/dataloader_idx_0"].detach().item(),
            trainer.callback_metrics["val_acc/dataloader_idx_1"].detach().item(),
            trainer.callback_metrics["val_acc/dataloader_idx_2"].detach().item(),
        ]
        return np.mean(val_acc)
    elif network_type == "mvcnn":
        return trainer.callback_metrics["val_acc"]


def main(args_cli):
    config = load_config(CONFIG_PATH)

    print(f'Number of views: {config["num_views"]}')

    # go to the parent directory
    parent_dir = Path(__file__).resolve().parents[2]
    if not config["lr_tuning"]:
        # sets seeds for numpy, torch and python.random.
        set_seed(21)

        for data_set in config["data_set_name"]:
            # go to the training data directory
            data_dir = parent_dir.joinpath("data", "raw", "Data", data_set)
            params = {"lr_optuna": config["learning_rate"]}

            for _ in range(1):
                with isolate_rng():
                    cnn_train(
                        parent_dir,
                        data_dir,
                        config,
                        config["network_type"],
                        data_set,
                        params=params,
                    )
    else:
        # Extra: This is for tuning the learning rate using Optuna and any other parameter.
        # sets seeds for numpy, torch and python.random.
        set_seed(21)
        for data_set in config["data_set_name"]:
            # go to the training data  in set1_clean directory
            data_dir = parent_dir.joinpath("data", "raw", "Data", data_set)
            # below with context keeps the seed after the with context same as before the with context
            with isolate_rng():

                def objective(trial):
                    params = {
                        "lr_optuna": trial.suggest_float(
                            "lr_optuna", 1e-4, 1e-2, log=True
                        )
                    }

                    val_acc = cnn_train(
                        parent_dir,
                        data_dir,
                        config,
                        config["network_type"],
                        data_set,
                        params=params,
                    )

                    return val_acc

                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=10)
                trail_ = study.best_trial
                print(trail_.values)
                print(trail_.params)


# Function to load yaml configuration file
def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default="gpu")
    args = parser.parse_args()

    main(args)
