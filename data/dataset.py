import torch
import torchvision.transforms as transforms
import os
import errno
import tarfile
from PIL import Image


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                                         #
#                You may change this file, but it is not necessary.                       #
#   For a better understanding of data preprocessing though we recommend reading it.      #
#                                                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class StrangeSymbols(torch.utils.data.Dataset):
    urls = [
        'http://ml.cs.uni-kl.de/download/strange_symbols.tar.gz',
    ]

    def __init__(self, root=None, train=True, transform=None):
        if root is None:
            root = os.path.join(os.path.dirname(__file__), '..', 'data')
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.transform = transform
        self.download()

        if self.train:
            self.train_data = torch.load(os.path.join(self.root, 'training_data.pt'))
            self.train_labels = torch.load(os.path.join(self.root, 'training_labels.pt'))
        else:
            self.test_data = torch.load(os.path.join(self.root, 'test_data.pt'))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img = self.test_data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return (img, target) if self.train else (img, None)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'training_data.pt')) and \
            os.path.exists(os.path.join(self.root, 'test_data.pt')) and \
            os.path.exists(os.path.join(self.root, 'training_labels.pt'))

    def download(self):
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)
            tarfile.open(out_f.name).extractall(self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str


def get_strange_symbol_loader(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (0.5, ))]
    )
    trainset = StrangeSymbols(train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    return trainloader


def get_strange_symbols_test_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, ), (0.5, ))]
    )

    return StrangeSymbols(
        train=False, transform=transform
    ).test_data[:, None, :, :].float().sub_(0.5).div_(0.5)


if __name__ == '__main__':
    trainloader = get_strange_symbol_loader(128)
    testdata = get_strange_symbols_test_data()
    print()
