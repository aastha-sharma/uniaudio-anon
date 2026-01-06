"""
Cross-Domain Dataset Wrapper

Adds domain labels to existing DeepfakeDataset for DANN training
"""
import pandas as pd
from torch.utils.data import Dataset
from src.data.dataset import DeepfakeDataset


class CrossDomainDeepfakeDataset(Dataset):
    """
    Wrapper that adds domain labels to DeepfakeDataset
    
    Domain labels: 0 = speech, 1 = singing
    """
    
    SPEECH_DATASETS = ['asvspoof2019', 'asvspoof2021', 'wavefake', 'in_the_wild']
    SINGING_DATASETS = ['svdd', 'singfake', 'ctrsvdd']
    
    def __init__(self, manifest_csv, split='train', **kwargs):
        """
        Args:
            manifest_csv: Path to manifest CSV
            split: 'train', 'dev', or 'test'
            **kwargs: Additional args for DeepfakeDataset
        """
        self.base_dataset = DeepfakeDataset(manifest_csv, split=split, **kwargs)
        
        # Load manifest to get dataset names
        self.df = pd.read_csv(manifest_csv)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        
        # Create domain label mapping
        self.domain_labels = []
        for dataset_name in self.df['dataset']:
            if dataset_name in self.SPEECH_DATASETS:
                self.domain_labels.append(0)  # Speech
            elif dataset_name in self.SINGING_DATASETS:
                self.domain_labels.append(1)  # Singing
            else:
                # Unknown dataset, default to speech
                self.domain_labels.append(0)
        
        assert len(self.domain_labels) == len(self.base_dataset), \
            "Domain labels length mismatch"
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get base dataset item
        base_item = self.base_dataset[idx]
        
        # base_item is tuple: (audio, label, dataset_name, ...)
        # Add domain label
        domain_label = self.domain_labels[idx]
        
        # Return (audio, label, domain_label, dataset_name, ...)
        if len(base_item) == 2:
            audio, label = base_item
            return audio, label, domain_label
        elif len(base_item) == 3:
            audio, label, dataset_name = base_item
            return audio, label, domain_label, dataset_name
        else:
            # Has more items, insert domain_label after label
            return base_item[:2] + (domain_label,) + base_item[2:]


def make_cross_domain_loaders(manifest_csv, batch_size=32, num_workers=4):
    """
    Create train/dev dataloaders with domain labels
    """
    from torch.utils.data import DataLoader
    
    train_dataset = CrossDomainDeepfakeDataset(
        manifest_csv,
        split='train',
        segment_sec=6,
        sr=16000
    )
    
    dev_dataset = CrossDomainDeepfakeDataset(
        manifest_csv,
        split='dev',
        segment_sec=6,
        sr=16000
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, dev_loader