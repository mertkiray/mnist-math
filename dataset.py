import os
import random
from typing import Tuple, Any

import torch
from PIL import Image
import torchvision.datasets
from torch.utils.data import DataLoader


class MNIST(torchvision.datasets.MNIST):
    def __init__(self, dataset_type='sumnist', *args, **kwargs):
        super(MNIST, self).__init__(*args, **kwargs)

        self.dataset_type = dataset_type

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, 'processed')

    # Limit data and labels for train and validation so they don't overlap.
    # This is not necessary for normal MNIST but because we are selecting a random data in getitem we have to manually
    # limit the data and labels.
    def set_indices(self, indices):
        self.data = self.data[indices]
        self.targets = self.targets[indices]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, image, target) where target is sum of the two target classes.
        """

        # Get image and label as normal
        img, target = self.data[index], int(self.targets[index])

        # This will select a random element from the dataset to be used as the pair image.
        random_index = random.randint(0, self.__len__() - 1)
        img2, target2 = self.data[random_index], int(self.targets[random_index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)

        # I choose to concat the images to create a (2,28,28) image that will be fed into the network.
        images = torch.cat((img, img2), dim=0)

        if self.target_transform is not None:
            target = self.target_transform(target)
            target2 = self.target_transform(target2)

        # If sumnist data is chosen, the label become label1 + label2.
        # We do not have to introduce the operation concept there because only sum can be made.
        if self.dataset_type == 'sumnist':
            target_sum = target + target2
            return images, target_sum

        # If diffsumnist is chosen, we have to introduce the operation concept.
        # I choose to introduce a random variable, which will be diff if probability <= 0.5 and sum else
        elif self.dataset_type == 'diffsumnist':
            operation_prob = random.random()
            if operation_prob <= 0.5:
                # operation is diff
                # The operation concept is given into the input directly. If the operation is diff, the third channel of
                # the image will be all zeros.
                images = torch.cat((images, torch.zeros(img.size())), dim=0)
                target_diff = abs(target - target2)
                return images, target_diff
            else:
                # operation is sum
                # If the operation is sum, the third channel of the image will be all ones.
                images = torch.cat((images, torch.ones(img.size())), dim=0)
                target_sum = target + target2
                return images, target_sum

    def __len__(self) -> int:
        return len(self.data)


def get_data_loaders(data_root, dataset_type, batch_size, transforms):
    train_dataset = MNIST(root=data_root, dataset_type=dataset_type, train=True, download=True, transform=transforms,
                          target_transform=None)
    val_dataset = MNIST(root=data_root, dataset_type=dataset_type, train=True, download=True, transform=transforms,
                        target_transform=None)
    test_dataset = MNIST(root=data_root, dataset_type=dataset_type, train=False, download=True, transform=transforms,
                         target_transform=None)

    dataset_size = len(train_dataset)

    # Partition dataset train, val
    train_size = int(dataset_size * 0.8)
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_dataset.set_indices(train_indices)
    val_dataset.set_indices(val_indices)

    print(f'Dataset size: {dataset_size}, train size: {train_size}, valid size: {dataset_size - train_size},'
          f' test size: {len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader
