#!/usr/bin/env python
# -*-coding:utf-8 -*-

# @File    :   client.py
# @Time    :   2023/01/21 11:36:46
# @Author  :   Alexandru-Andrei Iacob
# @Contact :   aai30@cam.ac.uk
# @Author  :   Lorenzo Sani
# @Contact :   ls985@cam.ac.uk, lollonasi97@gmail.com
# @Version :   1.0
# @License :   (C)Copyright 2023, Alexandru-Andrei Iacob, Lorenzo Sani
# @Desc    :   None

import numbers
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Tuple, Callable, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common.typing import NDArrays, Scalar
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from femnist_dataset import FEMNIST


# Load with appropriate transforms
def to_tensor_transform(p: Any) -> torch.Tensor:
    """Transform the object given to a PyTorch Tensor

    Args:
        p (Any): object to transform

    Returns:
        torch.Tensor: resulting PyTorch Tensor
    """
    return torch.tensor(p)


def load_FEMNIST_dataset(data_dir: Path, mapping: Path, name: str) -> Dataset:
    """Function to load the FEMNIST dataset given the mapping .csv file.
    The relevant transforms are automatically applied.

    Args:
        data_dir (Path): path to the dataset folder.
        mapping (Path): path to the mapping .csv file chosen.
        name (str): name of the dataset to load, train or test.

    Returns:
        Dataset: FEMNIST dataset object, ready-to-use.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    return FEMNIST(
        mapping=mapping,
        name=name,
        data_dir=data_dir,
        transform=transform,
        target_transform=to_tensor_transform,
    )


def train_FEMNIST(
    net: Module,
    train_loader: DataLoader,
    epochs: int,
    device: str,
    optimizer: torch.optim.Optimizer,
    criterion: Module,
    **kwargs,
) -> float:
    """Trains the network on the training set.

    Args:
        net (Module): generic module object describing the network to train.
        train_loader (DataLoader): dataloader to iterate during the training.
        epochs (int): number of epochs of training.
        device (str): device name onto which perform the computation.
        optimizer (torch.optim.Optimizer): optimizer object.
        criterion (Module): generic module describing the loss function.

    Returns:
        float: the final epoch mean train loss.
    """
    if "max_batches" in kwargs:
        max_batches = kwargs["max_batches"]
    else:
        max_batches = None
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


def test_FEMNIST(
    net: Module,
    test_loader: DataLoader,
    device: str,
    criterion: Module,
    **kwargs,
) -> Tuple[float, float]:
    """Validate the network on a test set.

    Args:
        net (Module): generic module object describing the network to test.
        test_loader (DataLoader): dataloader to iterate during the testing.
        device (str):  device name onto which perform the computation.
        criterion (Module): generic module describing the loss function.

    Returns:
        Tuple[float, float]: couple of average test loss and average accuracy on the test set.
    """
    if "max_batches" in kwargs:
        max_batches = kwargs["max_batches"]
    else:
        max_batches = None

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


##  Define a simple CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 62)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


##  Define a simple MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 62)

    def forward(self, x):
        x = self.flatten(x)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# All experiments will have the exact same initialization.
# All differences in performance will come from training
def get_network_generator_cnn():
    untrained_net: Net = Net()

    def generated_net():
        return deepcopy(untrained_net)

    return generated_net


# All experiments will have the exact same initialization.
# All differences in performance will come from training
def get_network_generator_mlp():
    untrained_net: MLP = MLP()

    def generated_net():
        return deepcopy(untrained_net)

    return generated_net


def set_model_parameters(net: Module, parameters: NDArrays) -> Module:
    """Function to put a set of parameters into the model object.

    Args:
        net (Module): model object.
        parameters (NDArrays): set of parameters to put into the model.

    Returns:
        Module: updated model object.
    """
    weights = parameters
    params_dict = zip(net.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    return net


def get_model_parameters(net: Module) -> NDArrays:
    """Function to get the current model parameters as NDArrays.

    Args:
        net (Module): current model object.

    Returns:
        NDArrays: set of parameters from the current model.
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def aggregate_weighted_average(metrics: List[Tuple[int, dict]]) -> dict:
    """Generic function to combine results from multiple clients
    following training or evaluation.

    Args:
        metrics (List[Tuple[int, dict]]): collected clients metrics

    Returns:
        dict: result dictionary containing the aggregate of the metrics passed.
    """
    average_dict: dict = defaultdict(list)
    total_examples: int = 0
    for num_examples, metrics_dict in metrics:
        for key, val in metrics_dict.items():
            if isinstance(val, numbers.Number):
                average_dict[key].append((num_examples, val))  # type:ignore
        total_examples += num_examples
    return {
        key: {
            "avg": float(
                sum([num_examples * metr for num_examples, metr in val])
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
) -> Callable[[int, NDArrays, Dict[str, Any]], Tuple[float, Dict[str, Scalar]]]:
    """Wrapper function for the external federated evaluation function.
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
        criterion (Module): PyTorch Module containing the criterion for evaluating the model.

    Returns:
        Callable[[int, NDArrays, Dict[str, Any]], Tuple[float, Dict[str, Scalar]]]: external federated evaluation function.
    """
    
    full_file: Path = centralized_mapping
    dataset: Dataset = load_FEMNIST_dataset(data_dir, full_file, "val")
    mean = np.mean([val[1] for val in dataset])
    num_samples = len(dataset)
    index_list = list(range(0, num_samples))
    prng = np.random.RandomState(1337)
    prng.shuffle(index_list)
    index_list = index_list[:1500]
    dataset =  torch.utils.data.Subset(dataset, index_list)

    print("Reduced federated test_set size from ", num_samples, " to a size of ", len(dataset), " mean index:", np.mean(index_list))

    def federated_evaluation_function(
        server_round: int,
        parameters: NDArrays,
        fed_eval_config: Dict[
            str, Any
        ],  # mandatory argument, even if it's not being used
    ) -> Tuple[float, Dict[str, Scalar]]:
        """Evaluation function external to the federation.
        It uses the centralized val set for sake of simplicity.

        Args:
            server_round (int): current federated round.
            parameters (NDArrays): current model parameters.
            fed_eval_config (Dict[str, Any]): mandatory argument in Flower, can contain some configuration info

        Returns:
            Tuple[float, Dict[str, Scalar]]: evaluation results
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


def get_default_train_config() -> Dict[str, Any]:
    return {
        "epochs": 8,
        "batch_size": 32,
        "client_learning_rate": 0.01,
        "weight_decay": 0.001,
        "num_workers": 0,
    }


def get_default_test_config() -> Dict[str, Any]:
    return {
        "batch_size": 32,
        "num_workers": 0,
    }

