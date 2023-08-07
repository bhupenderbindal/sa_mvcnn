from pathlib import Path
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, ConfusionMatrix

from src.models.plot_utilities import plot_mvcnn_batch
from src.models.cf_matrix import make_confusion_matrix


class MVCNNClassifier(pl.LightningModule):
    def __init__(self, model, lr=0.001, fusion_strategy="latefusion", plot_preds=False):
        """Manages train, validation, test loop and optimisers settings for MVCNN."""
        super().__init__()
        self.model = model
        self.lr = lr
        self.fusion_strategy = fusion_strategy
        self.plot_preds = plot_preds
        # save_hyperparameters saves all the provided parameters for the __init__
        self.save_hyperparameters(ignore=["model"])

        # set requires_grad=False for all parameters in the feature extractor
        for param in self.model.feature_extractor.parameters():
            param.requires_grad = False
        # print(self.model)
        self.test_confmat = []
        self.accuracy = Accuracy(task="multiclass", num_classes=4)
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=4)
        self.validation_confmat = []
        self.classification_lables = ["diamond", "gyroid", "lonsdaleite", "primitive"]
        self.training_step_loss = []
        self.val_step_loss = []

    def forward(self, x):
        return self.model.forward(x)

    def cross_entropy_loss(self, logits, labels):
        if self.fusion_strategy == "latefusion":
            return F.cross_entropy(logits, labels)
        elif (
            self.fusion_strategy == "scoremax"
            or self.fusion_strategy == "scoresum"
            or self.fusion_strategy == "scoremul"
        ):
            return F.nll_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        x, y, z = batch
        # print(x.shape, y.shape, len(z))
        # plot_mvcnn_batch(batch=batch)
        # print(
        #     f"x shape in training step of single sample i.e. numviews images : {x.shape}, label : {y}"
        # )
        y_hat = self.model(x)
        loss = self.cross_entropy_loss(y_hat, y)
        metrics = {"train_loss": loss}
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)
        # visualise train batch
        # plot_mvcnn_batch(batch=batch)
        self.training_step_loss.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, confmat = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)
        self.validation_confmat.append(confmat)
        # Extra: logs the accuracy to hparams tab in tensorboard
        # hp/<scalarname> logs the scalarname to the hparams
        self.log("hp/val_acc", acc)
        self.val_step_loss.append(loss)
        return metrics, confmat

    def test_step(self, batch, batch_idx):
        # plot_mvcnn_batch(batch=batch)
        loss, acc, confmat = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        self.test_confmat.append(confmat)
        return metrics, confmat

    def _shared_eval_step(self, batch, batch_idx):
        x, y, z = batch
        y_hat = self.model(x)
        loss = self.cross_entropy_loss(y_hat, y)
        if self.fusion_strategy == "latefusion":
            # Turn predicted logits in prediction probabilities
            y_hat = torch.softmax(y_hat, dim=1)
        elif (
            self.fusion_strategy == "scoremax"
            or self.fusion_strategy == "scoresum"
            or self.fusion_strategy == "scoremul"
        ):
            pass

        # Turn prediction probabilities into prediction labels
        y_preds = y_hat.argmax(dim=1)
        # accuracy = Accuracy(task="multiclass", num_classes=4)
        acc = self.accuracy(y_preds, y)
        confusion_matrix = self.confmat(preds=y_preds, target=y)

        # for the visualisation of the images, labels and predictions
        if self.plot_preds:
            plot_mvcnn_batch(batch=batch, preds=y_preds)
        return loss, acc, confusion_matrix

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # predict should get batch as only the image not the label

        x, _, _ = batch
        y_pred = self.model(x)
        if self.fusion_strategy == "latefusion":
            # Turn predicted logits in prediction probabilities
            y_pred = torch.softmax(y_pred, dim=1)
        elif (
            self.fusion_strategy == "scoremax"
            or self.fusion_strategy == "scoresum"
            or self.fusion_strategy == "scoremul"
        ):
            pass

        # Turn prediction probabilities or scores into prediction labels
        y_preds = y_pred.argmax(dim=1)

        plot_mvcnn_batch(batch=batch, preds=y_preds, inference=True)
        return y_preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Extra: use below to use lr scheduler with optimizer
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, verbose=True)
        # return {"optimizer": optimizer,"lr_scheduler": lr_scheduler}

    def conf_mat(self):
        # creates a conf mat for test
        a = torch.stack(self.test_confmat, axis=0).sum(axis=0, keepdim=True).squeeze()

        figure_confmat = make_confusion_matrix(
            cf=a.cpu().numpy(), categories=self.classification_lables
        )
        self.logger.experiment.add_figure(tag="confmat_test", figure=figure_confmat)
        # # reset to empty state
        self.test_confmat = []

    def on_validation_epoch_end(self):
        # Creates a confusion matrix for validation at every epoch end
        a = (
            torch.stack(self.validation_confmat, axis=0)
            .sum(axis=0, keepdim=True)
            .squeeze()
        )

        figure_confmat = make_confusion_matrix(
            cf=a.cpu().numpy(), categories=self.classification_lables
        )
        self.logger.experiment.add_figure(
            tag="confmat_validation", figure=figure_confmat
        )
        # reset to empty state
        self.validation_confmat.clear()

        # logs train and validation loss to same plot in tensorboard
        if bool(self.training_step_loss) and bool(self.val_step_loss):
            # do something with all training_step outputs, for example:
            train_loss_mean = torch.stack(self.training_step_loss).mean()
            val_loss_mean = torch.stack(self.val_step_loss).mean()

            self.logger.experiment.add_scalars(
                "train and val loss",
                {"train_losss": train_loss_mean, "val_losss": val_loss_mean},
                self.global_step,
            )
            # free up the memory
            self.training_step_loss.clear()
            self.val_step_loss.clear()

    def on_train_start(self):
        # Extra: logs the accuracy to hparams tab in tensorboard
        # hp/<scalarname> logs the scalarname to the hparams
        self.logger.log_hyperparams(
            self.hparams,
            {
                "hp/val_acc": -1,
                "hp/val_acc/dataloader_idx_0": -1,
                "hp/val_acc/dataloader_idx_1": -1,
                "hp/val_acc/dataloader_idx_2": -1,
            },
        )
