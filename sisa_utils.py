import os
import numpy as np
import torch
from torch.utils.data import Subset


class SISAManager:
    def __init__(self, dataset, model, seed, dataset_len, save_dir, num_shards, container_name=None):
        self.dataset = dataset
        self.model = model
        self.seed = seed
        self.dataset_len = dataset_len
        self.save_dir = save_dir
        self.num_shards = num_shards
        if container_name is None:
            container_name = f"{dataset}_{model}_{seed}"
        self.container_dir = os.path.join(save_dir, "containers", container_name)

        os.makedirs(self.container_dir, exist_ok=True)
        self.split_file = os.path.join(self.container_dir, "splitfile.npy")
        self.request_file = os.path.join(self.container_dir, "requestfile.npy")

    def partition_dataset(self):
        if os.path.exists(self.split_file):
            print(f"Loading existing partition from {self.split_file}")
            return np.load(self.split_file, allow_pickle=True)

        print(f"Creating new partition with {self.num_shards} shards")
        indices = np.arange(self.dataset_len)
        np.random.shuffle(indices)
        partition = np.array_split(indices, self.num_shards)

        partition_obj = np.array([np.array(p, dtype=int) for p in partition], dtype=object)
        np.save(self.split_file, partition_obj)

        requests = np.array([np.array([], dtype=int) for _ in range(self.num_shards)], dtype=object)
        np.save(self.request_file, requests)

        return partition_obj

    def add_unlearning_request(self, indices_to_unlearn):
        if not os.path.exists(self.request_file):
            raise FileNotFoundError("Run partition_dataset() first.")

        partition = np.load(self.split_file, allow_pickle=True)
        requests_arr = np.load(self.request_file, allow_pickle=True)
        requests_list = [np.asarray(shard_reqs, dtype=int) for shard_reqs in requests_arr]

        updated_shards = []
        indices_to_unlearn = np.asarray(indices_to_unlearn, dtype=int)
        for shard_id in range(self.num_shards):
            shard_indices = np.asarray(partition[shard_id], dtype=int)
            matches = np.intersect1d(shard_indices, indices_to_unlearn)

            if len(matches) > 0:
                current_reqs = requests_list[shard_id]
                new_reqs = np.union1d(current_reqs, matches).astype(int)
                requests_list[shard_id] = new_reqs
                updated_shards.append(shard_id)
                print(f"Shard {shard_id}: Added {len(matches)} indices to unlearn queue.")

        requests_final = np.array(requests_list, dtype=object)
        np.save(self.request_file, requests_final)

        return updated_shards

    def get_shard_loader(self, dataset, shard_id, slice_id, num_slices, batch_size):
        partition = np.load(self.split_file, allow_pickle=True)
        requests = np.load(self.request_file, allow_pickle=True)

        shard_indices = np.asarray(partition[shard_id], dtype=int)
        shard_requests = np.asarray(requests[shard_id], dtype=int)

        valid_indices = np.setdiff1d(shard_indices, shard_requests)

        total_valid = len(valid_indices)
        slice_size = total_valid // num_slices if num_slices > 0 else total_valid

        end_idx = (slice_id + 1) * slice_size
        if slice_id == num_slices - 1:
            end_idx = total_valid

        current_slice_indices = valid_indices[:end_idx]

        subset = Subset(dataset, current_slice_indices)

        return torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=4)

    def get_model_path(self, shard_id, slice_id):
        return os.path.join(
            self.container_dir,
            f"{self.dataset}_shard{shard_id}_slice{slice_id}_seed{self.seed}.pth",
        )

    def get_best_model_path(self):
        return os.path.join(self.container_dir, f"best_{self.dataset}_{self.model}_{self.seed}.pth")

    def get_history_path(self):
        return os.path.join(self.container_dir, f"history_{self.dataset}_{self.model}_{self.seed}.csv")

    def get_summary_path(self):
        return os.path.join(self.container_dir, f"summary_{self.dataset}_{self.model}_{self.seed}.txt")
