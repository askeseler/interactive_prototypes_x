import torchvision.transforms as transforms
import torchvision.datasets as dataset
import torchvision
import torch
from pathlib import Path
import pytorch_lightning as pl
from torchvision.datasets.folder import *
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import json
import os

debug = False

class ImageFolderWithGroundTruth(ImageFolder):
    """ An Image folder that returns the transformed images, as well as the path to the respective image files and
        the part labels upon call of __getitem__().
    """
    def __init__(self, root: str, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = default_loader, is_valid_file: Optional[Callable[[str], bool]] = None):
        """ Initializes the image folder with ground truth. The structure is exemplified by the following:

            dataset/groundtruth/train/class1
            dataset/groundtruth/train/class2
            dataset/images/train/class1
            dataset/imgaes/train/class2

        args:
            root: A path to the dataset/images/train/ or dataset/images/test/ folder. T
                  The grandparent directory must contain a folder groundtruth with the respective folder
                  substructure and identical file names for the labels.
            transforms: Transformations to be applied
            target_transform: Transformations to be applied to the targets
            loader: Loader instance
            is_valid_file: Callback for filtering valid files
            """
        super(torchvision.datasets.ImageFolder, self).__init__(root, loader,
                                          IMG_EXTENSIONS if is_valid_file is None else None, transform=transform,
                                          target_transform=target_transform, is_valid_file=is_valid_file)
        self.imgs_root = os.path.join(root)
        self.groundtruth_root = os.path.join(str(Path(root).parent.parent), "groundtruth", os.path.basename(self.imgs_root))
        self.groudtruth_paths = [s[0].replace("/images", "/groundtruth") for s in self.samples]
        self.groundtruth = None
        self.part_labels = None
        try:
            self.groundtruth = [np.array(Image.open(p).resize((7,7)))[:,:,0] / 255 for p in self.groudtruth_paths]
        except IOError as e:
            print("no groundtruth found")
        try:
            root_metafiles = os.path.join(str(Path(root).parent.parent))
            with open(os.path.expanduser(os.path.join(root_metafiles, "part_locs.json")), ) as f:
                self.part_labels = json.loads(f.read())
        except Exception as e:
            print("No part locs found")
            print(e)

    def __getitem__(self, index):
        """ Returns the item for the specified index. Each item contains the source image, the target_value,
            the path to the source_image and the path to the groundtruth image.
        Args:
            idx: Index of the sample
        Returns:
            extended_tuple: Tuple of the source image, the target_value, the path to the source_image and the path
            to the groundtruth image.
        """
        original_tuple = super(torchvision.datasets.ImageFolder, self).__getitem__(index)
        img_path = self.samples[index][0]
        groundtruth_path = self.groudtruth_paths[index]
        if type(self.groundtruth) != type(None):
            if type(self.part_labels) == type(None):
                extended_tuple = (original_tuple + (img_path, groundtruth_path, self.groundtruth[index]))
            else:                
                path = os.path.join(*self.samples[index][0].split("/")[-2:])#retrieve relative path
                if not path in self.part_labels:
                    print("missing value")
                    part_labels = torch.Tensor(self.part_labels[list(self.part_labels.keys())[0]])
                    part_labels.fill_(torch.nan)
                else:
                    part_labels = torch.Tensor(self.part_labels[path])
                extended_tuple = (original_tuple + (img_path, groundtruth_path, self.groundtruth[index], part_labels))
        else:
            extended_tuple = (original_tuple + (img_path, groundtruth_path,))
        return extended_tuple

class CUB200(pl.LightningDataModule):
    def __init__(self, debug = False, data_path = "~/datasets/cub200/images/", with_ground_truth = False):
        #data_path ='~/projects/interactive_learning_prototypes_experiments1/datasets/cub200_cropped/'
        if with_ground_truth:
            self.ImageFolder = ImageFolderWithGroundTruth
        else:
            self.ImageFolder = ImageFolder
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.preprocessing_function = lambda x: self.preprocess(x, mean=self.mean, std=self.std)
        self.undo_preprocessing = lambda x: self.undo_preprocessing(x, mean=self.mean, std=self.std)
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.debug = debug
        self.train_dir = data_path + 'train_cropped_augmented/'
        self.test_dir = data_path + 'test_cropped/'
        self.train_push_dir = data_path + 'train_cropped/'
        self.train_batch_size = 80
        self.test_batch_size = 100
        self.train_push_batch_size = 75
        self.img_size = 224
        self.init_datasets()

    def preprocess(self, x, mean, std):
        assert x.size(1) == 3
        y = torch.zeros_like(x)
        for i in range(3):
            y[:, i, :, :] = (x[:, i, :, :] - self.mean[i]) / std[i]
        return y

    def undo_preprocess(self, x, mean, std):
        assert x.size(1) == 3
        y = torch.zeros_like(x)
        for i in range(3):
            y[:, i, :, :] = x[:, i, :, :] * std[i] + self.mean[i]
        return y

    def init_datasets(self):
        # Datasets
        img_size = self.img_size
        train_dir, test_dir, train_push_dir = self.train_dir, self.test_dir, self.train_push_dir
        train_batch_size, test_batch_size, train_push_batch_size = self.train_batch_size, self.test_batch_size, self.train_push_batch_size
        self.train_dataset = self.ImageFolder(train_dir, transforms.Compose([transforms.Resize(size=(img_size, img_size)), transforms.ToTensor(), self.normalize, ]))
        self.train_push_dataset = self.ImageFolder(train_push_dir, transforms.Compose([transforms.Resize(size=(img_size, img_size)), transforms.ToTensor(), self.normalize, ]))
        self.train_push_dataset_raw = self.ImageFolder(train_push_dir, transforms.Compose([transforms.Resize(size=(img_size, img_size)), transforms.ToTensor(), ]))
        self.test_dataset = self.ImageFolder(test_dir, transforms.Compose([transforms.Resize(size=(img_size, img_size)), transforms.ToTensor(), self.normalize, ]))

        # Limit number of samples for debugging
        if self.debug:
           self.train_dataset.samples = self.train_dataset.samples[:100]
           self.train_push_dataset.samples = self.train_push_dataset.samples[:100]
           self.train_push_dataset_raw.samples = self.train_push_dataset_raw.samples[:100]
           self.test_dataset.samples = self.test_dataset.samples[:100]

        # Dataloaders
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=False)
        self.train_push_loader = torch.utils.data.DataLoader(self.train_push_dataset, batch_size=train_push_batch_size, shuffle=False, num_workers=4, pin_memory=False)
        self.train_push_loader_raw = torch.utils.data.DataLoader(self.train_push_dataset_raw, batch_size=train_push_batch_size, shuffle=False, num_workers=4, pin_memory=False)

    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.test_loader

    def train_push_dataloader(self):
        return self.train_push_loader

    def train_push_dataloader_raw(self):
        return self.train_push_loader_raw

class PlantNet(pl.LightningDataModule):
    def __init__(self, debug = False, data_path = "~/datasets/cub200/images/", with_ground_truth = False):
        #data_path ='~/projects/interactive_learning_prototypes_experiments1/datasets/cub200_cropped/'
        if with_ground_truth:
            self.ImageFolder = ImageFolderWithGroundTruth
        else:
            self.ImageFolder = ImageFolder
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.preprocessing_function = lambda x: self.preprocess(x, mean=self.mean, std=self.std)
        self.undo_preprocessing = lambda x: self.undo_preprocessing(x, mean=self.mean, std=self.std)
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.debug = debug
        self.train_dir = data_path + 'train_augmented/'
        self.test_dir = data_path + 'test/'
        self.train_push_dir = data_path + 'train/'
        self.train_batch_size = 80
        self.test_batch_size = 100
        self.train_push_batch_size = 75
        self.img_size = 224
        self.init_datasets()

    def preprocess(self, x, mean, std):
        assert x.size(1) == 3
        y = torch.zeros_like(x)
        for i in range(3):
            y[:, i, :, :] = (x[:, i, :, :] - self.mean[i]) / std[i]
        return y

    def undo_preprocess(self, x, mean, std):
        assert x.size(1) == 3
        y = torch.zeros_like(x)
        for i in range(3):
            y[:, i, :, :] = x[:, i, :, :] * std[i] + self.mean[i]
        return y

    def init_datasets(self):
        # Datasets
        img_size = self.img_size
        train_dir, test_dir, train_push_dir = self.train_dir, self.test_dir, self.train_push_dir
        train_batch_size, test_batch_size, train_push_batch_size = self.train_batch_size, self.test_batch_size, self.train_push_batch_size
        self.train_dataset = self.ImageFolder(train_dir, transforms.Compose([transforms.Resize(size=(img_size, img_size)), transforms.ToTensor(), self.normalize, ]))
        self.train_push_dataset = self.ImageFolder(train_push_dir, transforms.Compose([transforms.Resize(size=(img_size, img_size)), transforms.ToTensor(), self.normalize, ]))
        self.train_push_dataset_raw = self.ImageFolder(train_push_dir, transforms.Compose([transforms.Resize(size=(img_size, img_size)), transforms.ToTensor(), ]))
        self.test_dataset = self.ImageFolder(test_dir, transforms.Compose([transforms.Resize(size=(img_size, img_size)), transforms.ToTensor(), self.normalize, ]))

        # Limit number of samples for debugging
        if self.debug:
           self.train_dataset.samples = self.train_dataset.samples[:1000]
           self.train_push_dataset.samples = self.train_push_dataset.samples[:1000]
           self.train_push_dataset_raw.samples = self.train_push_dataset_raw.samples[:1000]
           self.test_dataset.samples = self.test_dataset.samples[:1000]

        # Dataloaders
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=False)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=False)
        self.train_push_loader = torch.utils.data.DataLoader(self.train_push_dataset, batch_size=train_push_batch_size, shuffle=False, num_workers=4, pin_memory=False)
        self.train_push_loader_raw = torch.utils.data.DataLoader(self.train_push_dataset_raw, batch_size=train_push_batch_size, shuffle=False, num_workers=4, pin_memory=False)

    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.test_loader

    def train_push_dataloader(self):
        return self.train_push_loader

    def train_push_dataloader_raw(self):
        return self.train_push_loader_raw
