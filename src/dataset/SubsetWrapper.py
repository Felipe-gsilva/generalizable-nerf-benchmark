from torch.utils.data import ConcatDataset, Dataset, Subset


class SubsetWrapper(Dataset):
    def __init__(self, subset: Subset, name: str, step: str):
        self.subset = subset
        self.dataset = subset.dataset
        self.name = name
        self.step = step

        if isinstance(self.dataset, ConcatDataset):
            # Pull metadata from the first constituent ImageFolder
            first_ds = self.dataset.datasets[0]
            self.classes = first_ds.classes
            self.class_to_idx = first_ds.class_to_idx
            self.dataset_path = first_ds.root

            # Build .samples / .targets by resolving each global index
            self.samples = []
            self.targets = []
            for i in subset.indices:
                ds_idx, local_idx = self._resolve_concat_index(i)
                child = self.dataset.datasets[ds_idx]
                self.samples.append(child.samples[local_idx])
                self.targets.append(child.targets[local_idx])
        else:
            self.classes = self.dataset.classes
            self.class_to_idx = self.dataset.class_to_idx
            self.dataset_path = self.dataset.root
            self.samples = [self.dataset.samples[i] for i in subset.indices]
            self.targets = [self.dataset.targets[i] for i in subset.indices]

    # ------------------------------------------------------------------
    def _resolve_concat_index(self, global_idx: int):
        """Map a ConcatDataset global index → (dataset_index, local_index)."""
        offset = 0
        for ds_idx, ds in enumerate(self.dataset.datasets):
            if global_idx < offset + len(ds):
                return ds_idx, global_idx - offset
            offset += len(ds)
        raise IndexError(f"Index {global_idx} out of range for ConcatDataset")

    def __getitem__(self, idx: int):
        return self.subset[idx]

    def __len__(self) -> int:
        return len(self.subset)
