"""Client utilities for the FEMNIST dataset."""

# @File    :   client.py
# @Time    :   2023/01/21 11:36:46
# @Author  :   Alexandru-Andrei Iacob
# @Contact :   aai30@cam.ac.uk
# @Author  :   Lorenzo Sani
# @Contact :   ls985@cam.ac.uk, lollonasi97@gmail.com
# @Version :   1.0
# @License :   (C)Copyright 2023, Alexandru-Andrei Iacob, Lorenzo Sani
# @Desc    :   None

import logging
import numbers
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, cast
from collections.abc import Callable, Sized

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from flwr.common.typing import NDArrays, Scalar
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from common.femnist_dataset import FEMNIST
from flwr.common.logger import log

class IntentionalDropoutError(BaseException):
    """For clients to intentionally drop out of the federated learning process."""

def get_device() -> str:
    """
    Get the device (CPU, CUDA, or MPS) available for computation.

    Returns
    -------
        str: The device available for computation.
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    return device


# Load with appropriate transforms
def to_tensor_transform(p: Any) -> torch.Tensor:
    """Transform the object given to a PyTorch Tensor.

    Args:
        p (Any): object to transform

    Returns
    -------
        torch.Tensor: resulting PyTorch Tensor
    """
    return torch.tensor(p)


def load_FEMNIST_dataset(  # noqa: N802
    data_dir: Path, mapping: Path, name: str
) -> Dataset:
    """Load the FEMNIST dataset given the mapping .csv file.

    The relevant transforms are automatically applied.

    Args:
        data_dir (Path): path to the dataset folder.
        mapping (Path): path to the mapping .csv file chosen.
        name (str): name of the dataset to load, train or test.

    Returns
    -------
        Dataset: FEMNIST dataset object, ready-to-use.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return FEMNIST(
        mapping=mapping,
        name=name,
        data_dir=data_dir,
        transform=transform,
        target_transform=to_tensor_transform,
    )


def train_FEMNIST(  # noqa: N802
    net: Module,
    train_loader: DataLoader,
    epochs: int,
    device: str,
    optimizer: torch.optim.Optimizer,
    criterion: Module,
    max_batches: int | None = None,
    **kwargs: dict[str, Any],
) -> float:
    """Trains the network on the training set.

    Args:
        net (Module): generic module object describing the network to train.
        train_loader (DataLoader): dataloader to iterate during the training.
        epochs (int): number of epochs of training.
        device (str): device name onto which perform the computation.
        optimizer (torch.optim.Optimizer): optimizer object.
        criterion (Module): generic module describing the loss function.

    Returns
    -------
        float: the final epoch mean train loss.
    """
    net.train()
    running_loss, total = 0.0, 0
    for _ in range(epochs):
        running_loss = 0.0
        total = 0
        batch_cnt = 0
        for data, labels in train_loader:
            if max_batches is not None and batch_cnt >= max_batches:
                break
            batch_cnt += 1
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(data), labels)
            running_loss += loss.item()
            total += labels.size(0)
            loss.backward()
            optimizer.step()
    return running_loss / total


def test_FEMNIST(  # noqa: N802
    net: Module,
    test_loader: DataLoader,
    device: str,
    criterion: Module,
    max_batches: int | None = None,
    **kwargs: dict[str, Any],
) -> tuple[float, float]:
    """Validate the network on a test set.

    Args:
        net (Module): generic module object describing the network to test.
        test_loader (DataLoader): dataloader to iterate during the testing.
        device (str):  device name onto which perform the computation.
        criterion (Module): generic module describing the loss function.

    Returns
    -------
        tuple[float, float]:
            couple of average test loss and average accuracy on the test set.
    """
    batch_cnt = 0
    correct, total, loss = 0, 0, 0.0
    net.eval()

    with torch.no_grad():
        for data, labels in tqdm(test_loader):

            if max_batches is not None and batch_cnt >= max_batches:
                break
            batch_cnt += 1

            data, labels = data.to(device), labels.to(device)
            outputs = net(data)

            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return loss, accuracy


def get_activations_from_random_input(
    net: Module,
    device: str,
    n_samples: int = 100,
    seed: int = 1337,
) -> np.ndarray:
    """Return the activations of the network on random input."""
    # Get a random input
    prng = torch.random.manual_seed(seed)
    random_input = torch.rand((n_samples, 1, 28, 28), generator=prng)
    random_input = random_input.to(device)
    # Get the activations
    net.to(device)
    net.eval()
    with torch.no_grad():
        outputs: torch.Tensor = torch.softmax(net(random_input), dim=1)
    average_activations = torch.mean(outputs, dim=0)
    return average_activations.cpu().numpy()


# Define a simple CNN
class Net(nn.Module):
    """Simple CNN model for FEMNIST."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 62)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns
        -------
            torch.Tensor: The output tensor.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define a simple MLP
class MLP(nn.Module):
    """Simple MLP model for FEMNIST."""

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 62)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns
        -------
            torch.Tensor: Output tensor.
        """
        x = self.flatten(x)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# All experiments will have the exact same initialization.
# All differences in performance will come from training
def get_network_generator_cnn() -> Callable[[], Net]:
    """Get function to generate a new CNN model."""
    untrained_net: Net = Net()

    def generated_net() -> Net:
        return deepcopy(untrained_net)

    return generated_net


# All experiments will have the exact same initialization.
# All differences in performance will come from training
def get_network_generator_mlp() -> Callable[[], MLP]:
    """Get function to generate a new MLP model."""
    untrained_net: MLP = MLP()

    def generated_net() -> MLP:
        return deepcopy(untrained_net)

    return generated_net


def set_model_parameters(net: Module, parameters: NDArrays) -> Module:
    """Get function to put a set of parameters into the model object.

    Args:
        net (Module): model object.
        parameters (NDArrays): set of parameters to put into the model.

    Returns
    -------
        Module: updated model object.
    """
    weights = parameters
    params_dict = zip(net.state_dict().keys(), weights, strict=False)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    return net


def get_model_parameters(net: Module) -> NDArrays:
    """Get function to get the current model parameters as NDArrays.

    Args:
        net (Module): current model object.

    Returns
    -------
        NDArrays: set of parameters from the current model.
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def aggregate_weighted_average(metrics: list[tuple[int, dict]]) -> dict:
    """Combine results from multiple clients.

    Args:
        metrics (list[tuple[int, dict]]): collected clients metrics

    Returns
    -------
        dict: result dictionary containing the aggregate of the metrics passed.
    """
    average_dict: dict = defaultdict(list)
    total_examples: int = 0
    for num_examples, metrics_dict in metrics:
        for key, val in metrics_dict.items():
            if isinstance(val, numbers.Number):
                average_dict[key].append((num_examples, val))
        total_examples += num_examples
    return {
        key: {
            "avg": float(
                sum([num_examples * m for num_examples, m in val])
                / float(total_examples)
            ),
            "all": val,
        }
        for key, val in average_dict.items()
    }


def get_federated_evaluation_function(
    data_dir: Path,
    centralized_mapping: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    model_generator: Callable[[], Module],
    criterion: Module,
) -> Callable[[int, NDArrays, dict[str, Any]], tuple[float, dict[str, Scalar]]]:
    """Wrap function for the external federated evaluation function.

    It provides the external federated evaluation function with some
    parameters for the dataloader, the model generator function, and
    the criterion used in the evaluation.

    Args:
        data_dir (Path): path to the dataset folder.
        centralized_mapping (Path): path to the mapping .csv file chosen.
        device (str):  device name onto which perform the computation.
        batch_size (int): batch size of the test set to use.
        num_workers (int): correspond to `num_workers` param in the Dataloader object.
        model_generator (Callable[[], Module]):  model generator function.
        criterion (Module): PyTorch Module containing the criterion.

    Returns
    -------
        Callable[[int, NDArrays, dict[str, Any]], tuple[float, dict[str, Scalar]]]:
            external federated evaluation function.
    """
    full_file: Path = centralized_mapping
    dataset: Dataset = load_FEMNIST_dataset(data_dir, full_file, "val")
    num_samples = len(cast(Sized, dataset))
    index_list = list(range(num_samples))
    prng = np.random.RandomState(1337)
    prng.shuffle(index_list)
    index_list = index_list[:1500]
    dataset = torch.utils.data.Subset(dataset, index_list)

    log(
        logging.INFO,
        "Reduced federated test_set size from %s to a size of %s mean index: %s",
        num_samples,
        len(cast(Sized, dataset)),
        np.mean(index_list),
    )

    def federated_evaluation_function(
        server_round: int,
        parameters: NDArrays,
        fed_eval_config: dict[
            str, Any
        ],  # mandatory argument, even if it's not being used
    ) -> tuple[float, dict[str, Scalar]]:
        """Evaluate on a centralized test set.

        It uses the centralized val set for sake of simplicity.

        Args:
            server_round (int): current federated round.
            parameters (NDArrays): current model parameters.
            fed_eval_config (dict[str, Any]): mandatory argument in Flower,
                                              can contain some configuration info

        Returns
        -------
            tuple[float, dict[str, Scalar]]: evaluation results
        """
        net: Module = set_model_parameters(model_generator(), parameters)
        net.to(device)

        valid_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

        loss, acc = test_FEMNIST(
            net=net,
            test_loader=valid_loader,
            device=device,
            criterion=criterion,
        )
        return loss, {"accuracy": acc}

    return federated_evaluation_function


def get_default_train_config() -> dict[str, Any]:
    """Get default training configuration."""
    return {
        "epochs": 8,
        "batch_size": 32,
        "client_learning_rate": 0.01,
        "weight_decay": 0.001,
        "num_workers": 0,
        "max_batches": 100,
    }


def get_default_test_config() -> dict[str, Any]:
    """Get default testing configuration."""
    return {
        "batch_size": 32,
        "num_workers": 0,
        "max_batches": 100,
    }
