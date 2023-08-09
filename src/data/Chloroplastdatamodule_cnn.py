import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader
from pathlib import Path
import torch
from torchvision import transforms

from src.data.chloroplast_dataset import (
    MultiViewChloroplastDataset,
    SingleViewChloroplastDataset,
)
from lightning.pytorch.utilities.combined_loader import CombinedLoader


class ChloroplastDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        num_views: int = 3,
        rand_rot_angle: tuple[int, int] = (0, 20),
        test_real_data=False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["data_dir"])

        self.data_dir = data_dir
        self.num_views = num_views
        self.rand_rot_angle = tuple(rand_rot_angle)
        self.test_real_data = test_real_data

        self.train_transform = transforms.Compose(
            [
                transforms.Resize(
                    232,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.RandomApply(
                    transforms=[
                        transforms.RandomRotation(degrees=self.rand_rot_angle),
                    ],
                    p=0.5,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
        self.test_pred_transform = transforms.Compose(
            [
                transforms.Resize(
                    232,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
            ]
        )

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train_dir = self.data_dir.joinpath("train")
            print("Reading training data from {}".format(self.train_dir))
            self.chloroplast_train = SingleViewChloroplastDataset(
                root_dir=self.train_dir, transform=self.train_transform
            )
            self.val_dir = self.data_dir.joinpath("evaluation")
            print("Reading validation data view-wise from {}".format(self.val_dir))

            self.chloroplast_val = MultiViewChloroplastDataset(
                self.val_dir,
                transform=self.test_pred_transform,
                num_views=self.num_views,
                view_num=1,
            ).svdataset()
            self.chloroplast_val_2 = MultiViewChloroplastDataset(
                self.val_dir,
                transform=self.test_pred_transform,
                num_views=self.num_views,
                view_num=2,
            ).svdataset()
            self.chloroplast_val_3 = MultiViewChloroplastDataset(
                self.val_dir,
                transform=self.test_pred_transform,
                num_views=self.num_views,
                view_num=3,
            ).svdataset()

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            if self.test_real_data:
                self.test_dir = self.data_dir.parents[0].joinpath("Real_data", "test")
                print("Reading real test data from {}".format(self.test_dir))
            else:
                self.test_dir = self.data_dir.joinpath("test")
                print("Reading test data from {}".format(self.test_dir))
            self.chloroplast_test = MultiViewChloroplastDataset(
                self.test_dir,
                transform=self.test_pred_transform,
                num_views=self.num_views,
                view_num=1,
            ).svdataset()
            self.chloroplast_test_2 = MultiViewChloroplastDataset(
                self.test_dir,
                transform=self.test_pred_transform,
                num_views=self.num_views,
                view_num=2,
            ).svdataset()
            self.chloroplast_test_3 = MultiViewChloroplastDataset(
                self.test_dir,
                transform=self.test_pred_transform,
                num_views=self.num_views,
                view_num=3,
            ).svdataset()

        if stage == "predict":
            self.predict_dir = self.data_dir.parents[0].joinpath(
                "Real_data", "no_class_mvcnn"
            )

            # this will load the images treating them as a single class for inference
            self.chloroplast_predict = MultiViewChloroplastDataset(
                self.predict_dir,
                transform=self.test_pred_transform,
                num_views=self.num_views,
                view_num=1,
            ).svdataset()
            self.chloroplast_predict_2 = MultiViewChloroplastDataset(
                self.predict_dir,
                transform=self.test_pred_transform,
                num_views=self.num_views,
                view_num=2,
            ).svdataset()
            self.chloroplast_predict_3 = MultiViewChloroplastDataset(
                self.predict_dir,
                transform=self.test_pred_transform,
                num_views=self.num_views,
                view_num=3,
            ).svdataset()

    def train_dataloader(self):
        return DataLoader(self.chloroplast_train, batch_size=32, shuffle=True)

    def val_dataloader(self):
        iterables = {
            "a": DataLoader(self.chloroplast_val, batch_size=32, shuffle=False),
            "b": DataLoader(self.chloroplast_val_2, batch_size=32, shuffle=False),
            "c": DataLoader(self.chloroplast_val_3, batch_size=32, shuffle=False),
        }
        combined_loader = CombinedLoader(iterables, "sequential")
        return combined_loader

    def test_dataloader(self):
        iterables = {
            "a": DataLoader(self.chloroplast_test, batch_size=32),
            "b": DataLoader(self.chloroplast_test_2, batch_size=32),
            "c": DataLoader(self.chloroplast_test_3, batch_size=32),
        }
        combined_loader = CombinedLoader(iterables, "sequential")
        return combined_loader

    def predict_dataloader(self):
        iterables = {
            "a": DataLoader(self.chloroplast_predict, batch_size=32),
            "b": DataLoader(self.chloroplast_predict_2, batch_size=32),
            "c": DataLoader(self.chloroplast_predict_3, batch_size=32),
        }
        combined_loader = CombinedLoader(iterables, "sequential")
        return combined_loader
