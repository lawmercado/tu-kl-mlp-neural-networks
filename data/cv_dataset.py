import torch
from torch.utils.data import Dataset, DataLoader


class RandomDataset(Dataset):
    def __init__(self, N, size):
        self.len = N
        self.data = torch.randn(N, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class CrossValidationDS(Dataset):
    """
    'Virtual' dataset that maps indexes to the real indexes in the original dataset,
    according to the given fold in k-folds-cross-validation.

    The original dataset is not copied multiple times in memory.
    """
    
    def __init__(self, ds, k, kv, train=True):
        assert kv >= 0 and kv < k, "kv should be in [0, k-1] but is {}".format(kv)

        self.ds = ds
        self.k = k
        self.kv = kv
        self.train = train

        self.fold_len = len(ds)//self.k
        if train:
            self.len = len(ds) - self.fold_len
        else:
            self.len = self.fold_len

    def index_to_real_ds_index(self, index):
        """
        Maps index of a splitted dataset to their positions in the original
        dataset.
        """
        if self.train:
            if index >= self.fold_len * self.kv:
                return index + self.fold_len
            return index
        else:
            return index + self.kv * self.fold_len

    def __getitem__(self, index):
        index = self.index_to_real_ds_index(index)
        return self.ds.__getitem__(index)

    def __len__(self):
        return self.len


def get_cv_datasets(ds, k):
    """
    For instance, k = 2 returns:
        [(ds_train_kv0, ds_val_kv0), (ds_train_kv1, ds_val_kv1)]
    """
    dss = []
    for kv in range(k):
        tset, vset = CrossValidationDS(ds, k, kv), CrossValidationDS(ds, k, kv, train=False)
        dss.append((tset, vset))

    return dss


def cv_datasets_to_dataloaders(cv_dss, batchsize, workers, shuffle_train=True):
    dls = []
    for tset, vset in cv_dss:
        tload = DataLoader(tset, batch_size=batchsize, shuffle=shuffle_train,
                           num_workers=workers)
        vload = DataLoader(vset, batch_size=batchsize, shuffle=False, 
                           num_workers=workers)
        dls.append((tload, vload))

    return dls


def test_cv_dataloaders():
    ds = RandomDataset(10, 1)
    cv_dss = get_cv_datasets(ds, 5)
    dls = cv_datasets_to_dataloaders(cv_dss, 1, 0)

    for epoch in range(4):
        print("Epoch: {}".format(epoch))

        for tload, vload in dls:
            print("Fold: {} of {}".format(tload.dataset.kv, tload.dataset.k-1))

            print("Train set:")
            for data in tload:
                print(data, end=',')

            print("\nVal set:")
            for data in vload:
                print(data, end=',')

            print('\n')


def test_get_cv_datasets_location():
    ds = RandomDataset(1000, 1)
    cv_dss = get_cv_datasets(ds, 5)

    for tset, vset in cv_dss:
        print("Fold: {} of {}".format(tset.kv, tset.k-1))
        print("Train DS: len {}, ds location {}".format(len(tset), hex(id(tset.ds))))
        print("Validation DS: len {}, ds location {}".format(len(vset), hex(id(vset.ds))))


def test_cv_datasets():
    ds = RandomDataset(10, 1)
    for i in range(len(ds)):
        print(ds[i], end=',')
    print('\n')

    cv_dss = get_cv_datasets(ds, 5)
    for tset, vset in cv_dss:
        print("Fold: {} of {}".format(tset.kv, tset.k-1))

        print("Train set:")
        for i in range(len(tset)):
            print(tset[i], end=',')

        print("\nVal set:")
        for i in range(len(vset)):
            print(vset[i], end=',')
        print('\n')


def test_shuffle():
    """
    Test whether shuffling one dataloader affects the other.

    Conclusion: the other dataloaders are not affected.
    
    """
    batchsize = 2
    ds = RandomDataset(10, 1)
    dl1 = DataLoader(ds, batchsize, shuffle=True)
    dl2 = DataLoader(ds, batchsize, shuffle=False)

    for epoch in range(3):
        print("Epoch: {}".format(epoch))

        for batch, (data1, data2) in enumerate(zip(dl1, dl2), 0):
            print("Batch: {}".format(batch))
            print("Data1: {}".format(data1))
            print("Data2: {}".format(data2))
            print()


def test_ds_location_in_dl():
    """
    Conclusion: all dataloaders share the same dataset.
    """
    batchsize = 2
    ds = RandomDataset(10, 1)
    dl1 = DataLoader(ds, batchsize, shuffle=True)
    dl2 = DataLoader(ds, batchsize, shuffle=False)

    print("Loc dl1: {}".format(loc(dl1.dataset)))
    print("Loc dl2: {}".format(loc(dl2.dataset)))


def loc(obj):
    return hex(id(obj))


if __name__ == "__main__":
    test_cv_dataloaders()