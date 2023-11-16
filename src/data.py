import os
import glob
import torch
import math
import numpy as np
import math

from torch.utils.data import Dataset
from tqdm import tqdm


class LabelTransformer:
    """
    Normalize label value to range -1 to 1
    """
    def __init__(self, probe_radius_upperbound: float, probe_radius_lowerbound: float):
        self.upperbound = probe_radius_upperbound
        self.lowerbound = probe_radius_lowerbound

    def transform(self, x: torch.Tensor):
        clamp_x = torch.clamp_(x, min=self.lowerbound, max=self.upperbound)
        trans_x = (clamp_x - self.lowerbound) / (self.upperbound - self.lowerbound) * 2 - 1
        return trans_x

    def inv_transform(self, x: torch.Tensor):
        inv_x = (x + 1) / 2 * (self.upperbound - self.lowerbound) + self.lowerbound
        return inv_x

    @property
    def sign_threshold(self):
        return self.transform(torch.zeros(1)).item()


class ExpLabelTransformer(LabelTransformer):
    """
    Normalize label value to range -1 to 1
    """
    def __init__(self, probe_radius_upperbound: float, probe_radius_lowerbound: float, offset: float):
        self.upperbound = probe_radius_upperbound
        self.lowerbound = probe_radius_lowerbound
        self.offset = offset

        self.exp_upperbound = math.exp(probe_radius_upperbound + self.offset)
        self.exp_lowerbound = math.exp(probe_radius_lowerbound + self.offset)

    def transform(self, x: torch.Tensor):
        clamp_x = torch.clamp_(x, min=self.lowerbound, max=self.upperbound)
        trans_x = torch.exp(clamp_x + self.offset)
        return trans_x

    def inv_transform(self, x: torch.Tensor):
        clamp_x = torch.clamp(x, min=self.exp_lowerbound, max=self.exp_upperbound)
        inv_x = torch.log(clamp_x) - self.offset
        return inv_x


class BipartScaleTransformer(LabelTransformer):
    """
    Normalize label value to range -1 to 1
    """
    def __init__(self, probe_radius_upperbound: float, probe_radius_lowerbound: float):
        self.upperbound = probe_radius_upperbound
        self.lowerbound = probe_radius_lowerbound

    def transform(self, x: torch.Tensor):
        clamp_x = torch.clamp_(x, min=self.lowerbound, max=self.upperbound)
        trans_x = torch.where(
            clamp_x > 0,
            clamp_x / abs(self.upperbound),
            clamp_x / abs(self.lowerbound)
        )
        return trans_x

    def inv_transform(self, x: torch.Tensor):
        inv_x = torch.where(
            x > 0,
            x * abs(self.upperbound),
            x * abs(self.lowerbound)
        )
        return inv_x


class TranslationLabelTransformer(LabelTransformer):
    """
    Normalize label value to range -1 to 1
    """
    def __init__(self, probe_radius_upperbound: float, probe_radius_lowerbound: float, do_truncate: bool = True):
        self.upperbound = probe_radius_upperbound
        self.lowerbound = probe_radius_lowerbound
        self.do_truncate = do_truncate

    def transform(self, x: torch.Tensor):
        if self.do_truncate:
            x = torch.clamp_(x, min=self.lowerbound, max=self.upperbound)
        trans_x = x - self.lowerbound
        return trans_x

    def inv_transform(self, x: torch.Tensor):
        inv_x = x + self.lowerbound
        return inv_x


class RefinedMlsesDataset(Dataset):
    """
    Load all data in dat files into the memory. Comsume too much memory.
    """
    def __init__(self, path, input_dim=200):
        self.path = path
        self.input_dim = input_dim
        self.dat_files = glob.glob(f'{path}/*/*.dat')

        print("Number of data files loading:", len(self.dat_files))

        self.labels = []
        self.features = []
        self.features_length = []
        for file in tqdm(self.dat_files):
            with open(file, "r") as f:
                for line in f:
                    row = line.split()

                    # for each sample, we have at least four elements
                    if len(row) > 3:
                        self.labels.append(float(row[0]))
                        feature = [float(i) for i in row[1:]]
                        self.features.append(feature)
                        self.features_length.append(len(feature))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        feature = self.features[index]
        feature_length = self.features_length[index]

        feature_array = np.array(feature, dtype=np.float32)
        feature_array = np.pad(feature_array, (0, self.input_dim - feature_length), 'constant')

        label_tensor = torch.LongTensor([label, ])
        feature_tensor = torch.from_numpy(feature_array)

        return feature_tensor, label_tensor


class RefinedMlsesMapDataset(Dataset):
    """
    Record entries offset inside the data file.
    """
    def __init__(self, dat_files, input_dim=200, label_transformer: LabelTransformer = None, label_only=False, dummy_columns=0):
        self.input_dim = input_dim
        self.dat_files = dat_files
        self.label_transformer = label_transformer
        self.label_only = label_only
        self.index_byte = 8
        self.dummy_columns = dummy_columns

        print("Number of data files loading:", len(self.dat_files))

        self.files_length = []
        for file in tqdm(self.dat_files):
            # check if offset index file has built
            index_file = self.__get_index_file(file)
            if os.path.isfile(index_file):
                with open(index_file, "rb") as f:
                    head_line = f.read(self.index_byte)
                    total_line = int.from_bytes(head_line, 'big')
                    self.files_length.append(total_line)
            else:
                # build offset index file
                with open(file, "r") as f:
                    offset = 0
                    file_offset = []
                    for line in f:
                        file_offset.append(offset)
                        offset += len(line)

                total_line = len(file_offset)
                self.files_length.append(total_line)

                with open(index_file, "wb") as f:
                    f.write(total_line.to_bytes(self.index_byte, 'big'))
                    for line_offset in file_offset:
                        f.write(line_offset.to_bytes(self.index_byte, 'big'))

        # build offset indexing
        self.files_cumsum = np.cumsum(self.files_length)
        print("Total line:", self.files_cumsum[-1])

    def __len__(self):
        return self.files_cumsum[-1].item()

    def __get_index_file(self, file):
        return f"{file}.offset_index"

    def get_file_name_by_index(self, index):
        file_index = np.searchsorted(self.files_cumsum, index, side='right')
        file_name = self.dat_files[file_index]
        infile_offset = index - (0 if file_index == 0 else self.files_cumsum[file_index - 1].item())

        # get index file and read offset
        index_file = self.__get_index_file(file_name)
        with open(index_file, 'rb') as f:
            f.seek(self.index_byte * (infile_offset + 1))  # +1 for skip head line
            line_offset = f.read(self.index_byte)
        file_offset = int.from_bytes(line_offset, 'big')
        return file_name, file_offset

    def __getitem__(self, index):
        # read line by offset
        file_name, file_offset = self.get_file_name_by_index(index)
        with open(file_name, 'r') as f:
            f.seek(file_offset)
            line = f.readline()

        # process line data
        if not self.label_only:
            row = line.split()
            label = float(row[0])
            # feature_num = math.log(len(row) - 1)
            feature = [float(i) for i in row[self.dummy_columns + 1:]]  # here we skip dummy columns

            # TODO: here is an extra colum for some training data files. Need to be fixed in the future.
            if len(feature) % 4 != 0:
                feature = feature[1:]
            assert len(feature) % 4 == 0, f'Feature len ({len(feature)}) cannot divided by 4.'

            # truncate feature
            feature = feature[:self.input_dim]

            feature_array = np.array(feature, dtype=np.float32, copy=False)
            feature_array = np.pad(feature_array, (0, self.input_dim - len(feature)), 'constant')

            label_tensor = torch.FloatTensor([label, ])
            feature_tensor = torch.from_numpy(feature_array)
        else:
            row = line.split()
            label = float(row[0])
            label_tensor = torch.FloatTensor([label, ])
            feature_num_tensor = torch.LongTensor([len(row) - 1, ])

        # do label transform if needed
        if self.label_transformer is not None:
            label_tensor = self.label_transformer.transform(label_tensor)

        if self.label_only:
            return feature_num_tensor, label_tensor
        else:
            return feature_tensor, label_tensor


class Subset(Dataset):
    def __init__(self, dataset: RefinedMlsesMapDataset, indices_path, split) -> None:
        self.dataset = dataset
        self.indices_file = f'{indices_path}/{split}.indices'

        self.length = 0
        with open(self.indices_file, 'rb') as f:
            head_line = f.read(self.dataset.index_byte)
            total_line = int.from_bytes(head_line, 'big')
            self.length = total_line

        print(f'{split} size:', self.length)

    def __getitem__(self, idx):
        with open(self.indices_file, 'rb') as f:
            f.seek(self.dataset.index_byte * (idx + 1))  # +1 for skip head line
            indice_bytes = f.read(self.dataset.index_byte)
        indice = int.from_bytes(indice_bytes, 'big')
        return self.dataset[indice]

    def __len__(self):
        return self.length


class RefinedMlsesMemoryMapDataset(Dataset):
    """
    Dataset from Numpy memmap
    """
    def __init__(self, dat_file, input_dim, sample_num, sample_dim, label_transformer: LabelTransformer = None, label_only=False):
        self.input_dim = input_dim
        self.sample_num = sample_num
        self.sample_dim = sample_dim
        self.dat_file = dat_file
        self.label_transformer = label_transformer
        self.label_only = label_only

        self.np_map = np.memmap(self.dat_file, dtype=np.float32, mode='r+', shape=(sample_num, sample_dim))
        print('Sample Num:', sample_num, 'Sample Dim:', sample_dim, 'Feature Size:', input_dim)

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        # read sample from memory map
        sample_row = self.np_map[index]
        feature_array = sample_row[:self.input_dim]
        label_array = sample_row[self.input_dim:]

        # process line data
        if not self.label_only:
            label_tensor = torch.from_numpy(label_array)
            feature_tensor = torch.from_numpy(feature_array)
        else:
            label_tensor = torch.from_numpy(label_array)
            feature_tensor = None

        # do label transform if needed
        if self.label_transformer is not None:
            label_tensor = self.label_transformer.transform(label_tensor)

        if self.label_only:
            return label_tensor
        else:
            return feature_tensor, label_tensor


class MultitaskRefinedMlsesMapDataset(RefinedMlsesMapDataset):
    def __init__(self, dat_files, input_dim=200, lowerbound=-1., label_transformer: LabelTransformer = None, label_only=False):
        super().__init__(dat_files, input_dim, label_transformer, label_only)
        self.input_dim = input_dim
        self.lowerbound = lowerbound

    def __getitem__(self, index):
        # read line by offset
        file_name, file_offset = self.get_file_name_by_index(index)
        with open(file_name, 'r') as f:
            f.seek(file_offset)
            line = f.readline()

        # process line data
        if not self.label_only:
            row = line.split()
            label = float(row[0])
            feature = [float(i) for i in row[1:]]

            # add the length as the first feature
            feature = [len(feature)] + feature

            # truncate feature
            feature = feature[:self.input_dim]

            feature_array = np.array(feature, dtype=np.float32)
            feature_array = np.pad(feature_array, (0, self.input_dim - len(feature)), 'constant')

            reg_label_tensor = torch.FloatTensor([label, ])
            cls_label_tensor = torch.LongTensor([1 if label < self.lowerbound else -1, ])  # 1 for trivial samples, -1 for nontrivial samples
            feature_tensor = torch.from_numpy(feature_array)
        else:
            row = line.split()
            label = float(row[0])
            reg_label_tensor = torch.FloatTensor([label, ])
            feature_tensor = None

        # do label transform if needed
        if self.label_transformer is not None:
            reg_label_tensor = self.label_transformer.transform(reg_label_tensor)

        if self.label_only:
            return reg_label_tensor
        else:
            return feature_tensor, reg_label_tensor, cls_label_tensor


class MultitaskRefinedMlsesMemoryMapDataset(RefinedMlsesMemoryMapDataset):
    def __init__(self, dat_file, input_dim, sample_num, sample_dim, lowerbound=-1., label_transformer: LabelTransformer = None, label_only=False):
        super().__init__(dat_file, input_dim, sample_num, sample_dim, None, label_only)
        self.input_dim = input_dim
        self.multitask_label_transformer = label_transformer
        self.sample_num = sample_num
        self.sample_dim = sample_dim
        self.lowerbound = lowerbound

    def __getitem__(self, index):
        if self.label_only:
            return super().__getitem__(index)
        else:
            feature_tensor, label_tensor = super().__getitem__(index)

            if self.multitask_label_transformer is not None:
                reg_label_tensor = self.multitask_label_transformer.transform(label_tensor)
            else:
                reg_label_tensor = label_tensor

            cls_label_tensor = torch.LongTensor([1 if label_tensor.item() < self.lowerbound else -1, ])  # 1 for trivial samples, -1 for nontrivial samples
            return feature_tensor, reg_label_tensor, cls_label_tensor


if __name__ == "__main__":
    dataset = RefinedMlsesDataset('dataset/benchmark_sample')
    print(len(dataset))
    print(dataset[200])
    print(dataset[200][0].shape)
