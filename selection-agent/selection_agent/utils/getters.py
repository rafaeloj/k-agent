from selection_agent.task import CNNModel, load_data
from selection_agent.tasks.models import CNNMNISTModel
from selection_agent.tasks.models import train_mnist_cnn, test_mnist_cnn
from selection_agent.tasks.datasets.load_mnist import load_mnist_data
from selection_agent.task import test as cnn_test_fn
from selection_agent.task import train as cnn_train_fn
import typing as T

def get_model(model_name: T.Literal["lstm", "cnn"], **kargs):
    if model_name == 'cnn':
        return CNNModel(), cnn_train_fn, cnn_test_fn
    if model_name == 'cnn-mnist':
        return CNNMNISTModel(), train_mnist_cnn, test_mnist_cnn

def get_dataset(partition_id: int, n_partitions: int, dataset_name: str, noniid:bool, **kargs):
    if dataset_name == 'cifar10':
        return load_data(partition_id=partition_id, num_partitions=n_partitions, noniid = noniid)
    if dataset_name == 'mnist':
        return load_mnist_data(partition_id=partition_id, num_partitions=n_partitions, noniid = noniid)