import torch
import torchvision.models as models
from torch import nn
from pathlib import Path


class CnnNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # initialize a pretrained model
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model to classify images in 4 classes
        num_target_classes = 4
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        # feature extractor is set to evaluation mode
        self.feature_extractor.eval()

        reprsentations = self.feature_extractor(x).flatten(1)
        # classifyin the representations
        x = self.classifier(reprsentations)
        return x


# model = CnnNetwork()


class MvCnnNetwork(nn.Module):
    def __init__(self, num_views=3, map_location="cpu"):
        super().__init__()
        self.num_views = num_views
        self.map_location = map_location

        parent_dir = Path(__file__).resolve().parents[2]

        # Add the path to the baseline trained models
        # Seperate baselines models can be used for training on
        # clean and noisy dataset

        # model_path = parent_dir.joinpath(
        #     "sv_baseline",
        #     "sv_cleandataset_baseline",
        #     "checkpoints",
        #     "epoch=26-step=2538.ckpt",
        # )

        model_path = parent_dir.joinpath(
            "sv_baseline",
            "sv_noisydataset_baseline",
            "checkpoints",
            "epoch=13-step=1316.ckpt",
        )

        checkpoint = torch.load(
            model_path, map_location=torch.device(self.map_location)
        )
        model = CnnNetwork()
        # Get the "model" state_dict
        state_dict = checkpoint["state_dict"]

        # Remove the "model" prefix from the keys in the state_dict
        state_dict = {
            key.replace("model.", ""): value for key, value in state_dict.items()
        }

        model.load_state_dict(state_dict)

        layers = list(model.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        num_filters = model.classifier.in_features

        # use the pretrained model to classify images in 4 classes
        num_target_classes = 4
        # as the features are concatenated together, input dimension is multiplied by num_views
        self.classifier = nn.Linear(num_filters * self.num_views, num_target_classes)

    def forward(self, x):
        # changing the (Batch, views, channels, length, width) to (v,b,c,l,w)
        x = x.transpose(0, 1)

        # creating an empty feature list
        features_list = []
        # feature extractor is set to evaluation mode
        self.feature_extractor.eval()

        for view in x:
            view_features = self.feature_extractor(view)
            features_list.append(view_features)

        # concatenate features_list
        concat_features_list = []
        for view_features in features_list:
            # reshaping the batch of extracted features from each view  as batch_size, flattend features
            flattend_features = view_features.view(view_features.shape[0], -1)
            concat_features_list.append(flattend_features)

        # concatenate the list feature wise
        concat_features = torch.concat(concat_features_list, 1)
        # print(
        #     f"concat feature shape: {concat_features.shape}, single feature shape: {num_filters}"
        # )
        #

        # classifyin the representations
        x = self.classifier(concat_features)
        return x


class MvCnnScoreNetwork(nn.Module):
    def __init__(self, num_views=3, fusion_strategy="scoresum", map_location="cpu"):
        super().__init__()
        self.num_views = num_views
        self.fusion_strategy = fusion_strategy
        self.map_location = map_location
        parent_dir = Path(__file__).resolve().parents[2]

        # Add the path to the baseline trained models
        # Seperate baselines models can be used for training on
        # clean and noisy dataset

        # model_path = parent_dir.joinpath(
        #     "sv_baseline",
        #     "sv_cleandataset_baseline",
        #     "checkpoints",
        #     "epoch=26-step=2538.ckpt",
        # )

        model_path = parent_dir.joinpath(
            "sv_baseline",
            "sv_noisydataset_baseline",
            "checkpoints",
            "epoch=13-step=1316.ckpt",
        )

        checkpoint = torch.load(
            model_path, map_location=torch.device(self.map_location)
        )
        model = CnnNetwork()
        # Get the "model" state_dict
        state_dict = checkpoint["state_dict"]

        # Remove the "model" prefix from the keys in the state_dict
        state_dict = {
            key.replace("model.", ""): value for key, value in state_dict.items()
        }

        model.load_state_dict(state_dict)

        layers = list(model.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        num_filters = model.classifier.in_features

        # use the pretrained model to classify images in 4 classes
        num_target_classes = 4
        # Num of filters or features remain same as single view in score fusion
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        # changing the (Batch, views, channels, length, width) to (v,b,c,l,w)
        x = x.transpose(0, 1)

        # creating an empty featuer list
        scores_list = []
        # feature extractor is set to evaluation mode
        self.feature_extractor.eval()

        for view in x:
            view_features = self.feature_extractor(view)
            flattend_features = view_features.view(view_features.shape[0], -1)
            x = self.classifier(flattend_features)
            # the scores are calculated as logrithmic of softmax
            x = nn.LogSoftmax(dim=-1)(x)

            scores_list.append(x.unsqueeze(dim=0))

        # # concatenate the score list view wise
        concat_scores = torch.concat(scores_list, 0)

        if self.fusion_strategy == "scoremax":
            scores = torch.max(concat_scores, 0).values
        elif self.fusion_strategy == "scoresum":
            scores = torch.sum(concat_scores, 0)
        elif self.fusion_strategy == "scoremul":
            scores = torch.prod(concat_scores, 0)

        # classifyin the representations
        return scores


# model = MvCnnNetwork()
# print(model)
