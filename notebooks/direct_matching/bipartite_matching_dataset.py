import os
import os.path as osp
import re

import torch
from torch_geometric.data import Data, Dataset

import clrs


class BipartiteMatchingDataset(Dataset):
    def __init__(self, root, num_samples, graph_size, transform = None, pre_transform = None, pre_filter = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.num_samples = num_samples
        self.graph_size = graph_size

    @property
    def raw_file_names(self):
        """If these files are found in the raw directory, download is skipped"""
        return []

    @property
    def processed_file_names(self):
        """If these files are found in the processed directory, processing is skipped"""
        processed_files = []
        path = self.processed_dir
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if not os.path.isdir(file_path) and not file == "pre_filter.pt" and not file == "pre_transform.pt":
                processed_files.append(file)

        return processed_files

    def download(self):
        pass

    def _iterate_sampler(self, sampler):
        while True:
            yield sampler.next()

    def process(self):
        # Start by getting the last file id to avoid overwriting (since we expect
        # filenames to be formatted as "data_{id}.pt")
        idx = 0
        found_file = False
        for file in os.listdir(self.processed_dir):
            prefix_string = "data_"
            if file.startswith(prefix_string):
                found_file = True
                file_id = int(re.findall(r'\d+', file)[0])
                idx = max(idx, file_id)
        if found_file:
            idx += 1

        sampler, _ = clrs.build_sampler(
            name = 'auction_matching',
            num_samples = self.num_samples,
            length = self.graph_size,
            weighted = True)  # number of nodes

        sampler = self._iterate_sampler(sampler)

        for i in range(self.num_samples):
            graph = next(sampler)



        path = self.raw_dir
        for file in os.listdir(path):
            print(file)

            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                continue
            # Read data from `raw_path`.
            output = process_file(file_path, 'nx')
            if output["converged"]:
                x = output["x"].type(torch.FloatTensor)
                edge_index = output["edge_index"]
                edge_attr = output["edge_attr"].type(torch.FloatTensor)
                y = output["y"]
                data = Data(x = x, edge_index = edge_index, edge_attr = edge_attr, y = y, filename = file)

                torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
                idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
