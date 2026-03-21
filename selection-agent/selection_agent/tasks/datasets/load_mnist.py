from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

pytorch_transforms = Compose([ToTensor()])


def apply_transforms_mnist(batch):
    batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
    return batch

def load_mnist_data(partition_id: int, num_partitions: int, noniid: bool = False):
    """Load partition MNIST data."""        
    partitioner = IidPartitioner(num_partitions=num_partitions) if not noniid else DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by="label",
        alpha=0.5,
        min_partition_size=10,
        self_balancing=True
    )
    fds = FederatedDataset(
        dataset='mnist',
        partitioners={"train": partitioner},
    )
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test = partition_train_test.with_transform(apply_transforms_mnist)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader