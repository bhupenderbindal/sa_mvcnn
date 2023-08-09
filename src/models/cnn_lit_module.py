from pathlib import Path
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, ConfusionMatrix

from src.models.plot_utilities import plot_cnn_batch
from src.models.cf_matrix import make_confusion_matrix


class CNNClassifier(pl.LightningModule):
    def __init__(self, model, lr=0.001, plot_preds=False):
        """Manages train, validation, test loop and optimisers settings for single view CNN."""
        super().__init__()
        self.model = model
        self.lr = lr
        self.plot_preds = plot_preds

        # save_hyperparameters saves all the provided parameters for the __init__
        self.save_hyperparameters(ignore=["model"])

        # set requires_grad=False for all parameters in the feature extractor before "feature_extractor.5" layer
        for name, param in self.model.named_parameters():
            if ("feature_extractor.5") > name:
                param.requires_grad = False
            else:
                if param.requires_grad:
                    pass
                    # print(param.numel())
                    # print(f" {name} layer is not frozen")
        # uncomment the following line to see the name of all the layers in the model
        # print(self.model)
        self.test_confmat = []
        self.accuracy = Accuracy(task="multiclass", num_classes=4)
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=4)
        self.validation_confmat = []

        self.classification_lables = ["diamond", "gyroid", "lonsdaleite", "primitive"]

    def forward(self, x):
        print(
            "forward in lightning module, can be used to do some extra modifications on top of the model.forward function"
        )
        return self.model.forward(x)

    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # plot_cnn_batch(batch=batch) # in case one wants to visualise the input data to network

        y_hat = self.model(x)
        loss = self.cross_entropy_loss(y_hat, y)
        metrics = {"train_loss": loss}
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        loss, acc, confmat = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, prog_bar=True, on_epoch=True, on_step=False)
        self.validation_confmat.append(confmat)
        # Extra: logs the accuracy to hparams tab in tensorboard
        # hp/<scalarname> logs the scalarname to the hparams
        self.log("hp/val_acc", acc)
        return metrics, confmat

    def test_step(self, batch, batch_idx, dataloader_idx):
        print(f" View {dataloader_idx+1} predictions")
        loss, acc, confmat = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        self.test_confmat.append(confmat)
        return metrics, confmat

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.cross_entropy_loss(y_hat, y)

        # Turn predicted logits in prediction probabilities
        y_pred_probs = torch.softmax(y_hat, dim=1)

        # Turn prediction probabilities into prediction labels
        y_preds = y_pred_probs.argmax(dim=1)

        # for the visualisation of the images, labels and predictions
        if self.plot_preds:
            plot_cnn_batch(batch=batch, preds=y_preds)

        acc = self.accuracy(y_preds, y)
        confusion_matrix = self.confmat(preds=y_preds, target=y)
        return loss, acc, confusion_matrix

    def predict_step(self, batch, batch_idx, dataloader_idx):
        # predict should get batch as only the image not the label
        print(f" View {dataloader_idx+1} predictions")

        x, _ = batch
        y_pred = self.model(x)

        # Turn predicted logits in prediction probabilities
        y_pred_probs = torch.softmax(y_pred, dim=1)

        # print the max probabilities for each prediction
        # print(y_pred_probs.max())

        # Turn prediction probabilities into prediction labels
        y_preds = y_pred_probs.argmax(dim=1)

        plot_cnn_batch(batch, y_preds, inference=True)
        return y_preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

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

    #  https://lightning.ai/docs/pytorch/latest/extensions/logging.html?highlight=hp_metric#logging-hyperparameters
    # Using custom or multiple metrics (default_hp_metric=False)
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
