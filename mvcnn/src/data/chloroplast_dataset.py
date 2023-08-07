from torchvision import datasets
import torch
from pathlib import Path
from typing import Any, Tuple


class SingleViewChloroplastDataset(datasets.ImageFolder):
    """Creates a single view chloroplast dataset from a directory having the following structure:
    root/diamond/xxx.png
    root/diamond/xxy.png

    root/gyroid/123.png
    root/gyroid/nsdf3.png
    """

    def __init__(self, root_dir: str, transform):
        self.root_dir = root_dir
        self.transform = transform

        super().__init__(
            root=self.root_dir, transform=self.transform, target_transform=None
        )


class MultiViewChloroplastDataset(datasets.ImageFolder):
    """Creates a multi view chloroplast dataset, by grouping images in Lexicographic order, from a directory having the following structure:
    root/diamond/diamond_00000_0.png
    root/diamond/diamond_00000_1.png
    root/diamond/diamond_00000_2.png

    root/gyroid/gyroid_00000_0.png
    root/gyroid/gyroid_00000_1.png
    root/gyroid/gyroid_00000_2.png

    """

    def __init__(self, root_dir: str, transform, num_views: int = 3, view_num: int = 1):
        self.root_dir = root_dir
        self.num_views = num_views
        self.transform = transform
        self.view_num = view_num

        super().__init__(
            root=self.root_dir, transform=self.transform, target_transform=None
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, path) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path

    def mvdataset(self):
        """
        mvdataset generates a MV dataset by first segregating the images by class
        and the grouping the images together as per num of views. The images are grouped by
        Lexicographic order of the image file names with the assumption that equal number of views are available for each MV sample.

        Returns
        -------
        list
            Returns list of tuples(mvimage,class, path)
        """

        print(f"total number of images: {len(self.samples)}")
        # print(self.find_classes(self.root_dir))
        # (['diamond', 'gyroid', 'lonsdaleite', 'primitive'], {'diamond': 0, 'gyroid': 1, 'lonsdaleite': 2, 'primitive': 3})
        mv_dict = {}

        # create a dictionary {label: list[(image, path)]}
        for index in range(len(self.samples)):
            image, label, path = self.__getitem__(index)
            if label not in mv_dict:
                mv_dict[(label)] = []
            mv_dict[(label)].append([image, path])

        mv_data_list = []
        # now for each label, grouping the views and create a MV image
        for key, value in mv_dict.items():
            label = key
            # value is a tuple of image and its path
            # as the images are named with view prefix, the grouping of images is done in that order by default
            image_list_by_class = value

            length_by_class = len(image_list_by_class)
            length_as_view_multiple = length_by_class - length_by_class % self.num_views

            for i in range(0, length_as_view_multiple, self.num_views):
                # selecting num_views tuples for each class
                mv_image_path_list = image_list_by_class[i : i + self.num_views]
                # then create list of images and paths from selected num_views samples
                mv_image_list = [item[0] for item in mv_image_path_list]
                mv_path = [item[1][-45:] for item in mv_image_path_list]
                mv_path = "\n".join(mv_path)

                mv_image = torch.stack(mv_image_list)
                mv_data_list.append((mv_image, label, mv_path))

        print(f"total samples in multi-view data after grouping: {len(mv_data_list)}")

        return mv_data_list

    def svdataset(self):
        """Returns the view wise dataset for the given view_num."""
        view_num = self.view_num
        sv_ds = []

        # creates a single view dataset based on view number
        for sample_ in self.mvdataset():
            sv_ds.append((sample_[0][view_num - 1].squeeze(), sample_[1]))

        return sv_ds
