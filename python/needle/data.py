import numpy as np
import gzip
import idx2numpy
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
            return img[:, ::-1, :]
        else:
            return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=0, high=self.padding * 2+1, size=2)
        print("shape:", img.shape)
        print("padding", self.padding)
        H, W, C = img.shape
        full_img = np.pad(img, self.padding)
        return full_img[shift_x: shift_x + H, shift_y: shift_y + W, self.padding:self.padding + C]


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

        seq_index = np.arange(len(dataset))
        if self.shuffle:
            np.random.shuffle(seq_index)
        self.ordering = np.array_split(seq_index, 
            range(batch_size, len(dataset), batch_size))
        self.m = len(self.ordering)
        self.i = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.m:
            raise StopIteration
        else:
            # print("self.i = ", self.i)
            # print("order[i] = ", self.ordering[self.i])
            # print("self.dataset = ", self.dataset)
            res = self.dataset[self.ordering[self.i]]
            # print("res", len(res), res[0].shape)
            self.i+=1
            return [Tensor(t) for t in res]


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        super().__init__(transforms)
        with gzip.open(image_filename) as f:
            X = idx2numpy.convert_from_file(f)
            X = X.reshape(-1, 28, 28, 1)
            X = X / 255.0
        self.X = X.astype('float32')
        with gzip.open(label_filename) as f:
            y = idx2numpy.convert_from_file(f)
            y = y.reshape(-1)
        self.y = y.astype('uint8')


    def __getitem__(self, index) -> object:
        return self.apply_transforms(self.X[index, :]), self.y[index]

    def __len__(self) -> int:
        return self.X.shape[0]


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
