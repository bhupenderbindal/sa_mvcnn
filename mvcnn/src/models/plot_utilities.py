import random

import matplotlib.pyplot as plt
import torchvision
from torchvision.utils import make_grid
import torch
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from captum.attr import GradientShap
from captum.attr import visualization as viz

font = {
    "family": "serif",
    "weight": "normal",
    "size": 16,
}


def grid_maker(images):
    grid = make_grid(images.squeeze())
    return grid


def plot_mvcnn_batch(batch, preds=None, inference=False):
    # this function can be called inside train, test, share_eval  methods in train_mvcnn
    images_batch, labels_batch, paths_batch = batch

    if preds is not None:
        for mv_image, mv_label, mv_path, mv_pred in zip(
            images_batch, labels_batch, paths_batch, preds
        ):
            grid = make_grid(mv_image)
            img = torchvision.transforms.ToPILImage()(grid)

            plt.imshow(img)
            plt.suptitle(
                "multi view CNN: Views of the sample", fontdict=font, color="C0"
            )

            label_list = mv_label.tolist()
            str_label = "label class: " + str(label_list)
            if inference:
                # labels in inference mode are just the num of folders
                str_label = "Inference mode"
                plt.suptitle("Inference: multi view CNN", fontdict=font, color="C0")
            if preds is not None:
                pred_list = mv_pred.tolist()
                str_pred = str(pred_list)
                str_label = str_label + "\n pred class:" + str_pred

            plt.xticks(ticks=[])
            plt.yticks(ticks=[])
            plt.xlabel(str_label + "\n path of images: \n" + str(mv_path))
            plt.show()

    else:
        for mv_image, mv_label, mv_path in zip(images_batch, labels_batch, paths_batch):
            grid = make_grid(mv_image)
            img = torchvision.transforms.ToPILImage()(grid)

            plt.imshow(img)
            plt.suptitle(
                "multi view CNN: Views of the sample", fontdict=font, color="C0"
            )

            label_list = mv_label.tolist()
            str_label = "label class: " + str(label_list)
            if inference:
                # labels in inference mode are just the num of folders
                str_label = "Inference mode"
                plt.suptitle("Inference: multi view CNN", fontdict=font, color="C0")

            plt.xticks(ticks=[])
            plt.yticks(ticks=[])
            plt.xlabel(str_label + "\n path of images: \n" + str(mv_path))
            plt.show()


def plot_cnn_batch(batch, preds=None, inference=False):
    # plot a batch of images, labels and predictions
    # inference is True for predictions ie plot of only images and preds
    # this function can be called inside train, share_eval  methods in train_cnn.py
    images, labels = batch

    # for mv_image, mv_label in zip(images_batch, labels_batch):
    grid = make_grid(images, nrow=8)
    img = torchvision.transforms.ToPILImage()(grid)

    plt.imshow(img)
    # locs, labels = plt.xticks()

    plt.suptitle("single view CNN", fontdict=font, color="C0")

    label_list = labels.tolist()
    str_label = [str(i) for i in label_list]
    str_label = "label class: " + "-".join(str_label)
    if inference:
        str_label = "Inference mode"
        plt.suptitle(
            "View-wise inference for single view CNN", fontdict=font, color="C0"
        )

    if preds is not None:
        pred_list = preds.tolist()
        str_pred = [str(i) for i in pred_list]
        str_pred = "-".join(str_pred)
        str_label = str_label + "\n pred class:" + str_pred

    plt.xticks(ticks=[])
    plt.yticks(ticks=[])
    plt.xlabel(str_label)
    plt.show()


def captum_grad_shap(batch, model, network_type="cnn", fusion_strategy="latefusion"):
    images_labels_paths = batch
    if network_type == "cnn":
        images_labels_paths = batch[0]

    random_select = random.randint(0, len(batch))
    input_ = images_labels_paths[0][random_select]

    true_label = images_labels_paths[1][random_select]
    if network_type == "mvcnn":
        print(f"paths: {images_labels_paths[2][random_select]}")

    default_cmap = LinearSegmentedColormap.from_list(
        "custom blue", [(0, "#ffffff"), (0.25, "#000000"), (1, "#000000")], N=256
    )

    input_ = input_.unsqueeze(0)
    output = model(input_)

    if network_type == "mvcnn":
        if fusion_strategy == "latefusion":
            # Turn predicted logits in prediction probabilities
            output = torch.nn.functional.softmax(output, dim=1)
        elif (
            fusion_strategy == "scoremax"
            or fusion_strategy == "scoresum"
            or fusion_strategy == "scoremul"
        ):
            pass

    elif network_type == "cnn":
        # Turn predicted logits in prediction probabilities
        output = torch.nn.functional.softmax(output, dim=1)

    prediction_score, pred_label_idx = torch.topk(output, 1)

    label_str = (
        "\n true label: "
        + str(
            true_label.squeeze().item(),
        )
        + "\n predicted label: "
        + str(
            pred_label_idx.squeeze().item(),
        )
    )
    gradient_shap = GradientShap(model)

    # Defining baseline distribution of images
    rand_img_dist = torch.cat([input_ * 0, input_ * 1])

    # baseline from random inputs
    # interclass_baselines = torch.cat(
    #     [
    #         images_labels_paths[0][random_select + 1].unsqueeze(0),
    #         images_labels_paths[0][random_select + 4].unsqueeze(0),
    #     ]
    # )
    attributions_gs = gradient_shap.attribute(
        input_,
        n_samples=40,
        stdevs=0.0001,
        baselines=rand_img_dist,
        target=pred_label_idx,
    )
    if network_type == "mvcnn":
        attributions_gs = grid_maker(attributions_gs)
        input_ = grid_maker(input_)

    _ = viz.visualize_image_attr_multiple(
        np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        np.transpose(input_.squeeze().cpu().detach().numpy(), (1, 2, 0)),
        ["original_image", "heat_map", "blended_heat_map"],
        ["all", "absolute_value", "absolute_value"],
        titles=["original_image" + label_str, "heat_map", "blended_heat"],
        cmap=default_cmap,
        show_colorbar=True,
    )
