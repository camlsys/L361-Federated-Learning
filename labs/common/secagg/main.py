"""Python module to run the secure aggregation demo."""

from enum import IntEnum
import random
from typing import Any
from collections.abc import Callable
from pathlib import Path
import os
from datetime import datetime, timezone
import json
import flwr as fl
from flwr.common import NDArrays
from flwr.client.client import Client
from flwr.server.strategy import Strategy
from flwr.server import ServerConfig, History
import numpy as np
import torch

from common.secagg.client import get_sec_agg_client_generator
from common.secagg.strategy import SecureAggregationStrategy

PARAMETERS = {
    "num_clients_per_round": 10,
    "num_total_clients": 100,
    "min_num_surviving_clients": 7,
    "num_dropouts": 3,
    "num_rounds": 4,
    # TODO: check consistency with folder tree
    "data_dir": "./client_data",
    "n_samples": 10,
    "n_dim": 10000,
}

HOME_DIR = Path("/content") if (content := Path("/content")).exists() else Path.cwd()


class Seeds(IntEnum):
    """Class for dealing with seeds."""

    DEFAULT = 1337


def convert(o: Any) -> int | float:
    """Convert input object to Python numerical if numpy."""
    # type: ignore[reportGeneralTypeIssues]
    if isinstance(o, np.int32 | np.int64):
        return int(o)
    # type: ignore[reportGeneralTypeIssues]
    if isinstance(o, np.float32 | np.float64):
        return float(o)
    raise TypeError


def save_history(history: History, name: str, home_dir: Path = HOME_DIR) -> None:
    """Save history from simulation to file."""
    time = int(datetime.now(timezone.utc).timestamp())
    path = home_dir / "histories"
    path.mkdir(exist_ok=True)
    path = path / f"hist_{time}_{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history.__dict__, f, ensure_ascii=False, indent=4, default=convert)


def start_seeded_simulation(
    client_fn: Callable[[str], Client],
    num_clients: int,
    config: ServerConfig,
    strategy: Strategy,
    name: str,
    seed: int = Seeds.DEFAULT,
    iteration: int = 0,
) -> tuple[list[tuple[int, NDArrays]], History]:
    """Wrap simulation to always seed client selection."""
    np.random.seed(seed ^ iteration)
    torch.manual_seed(seed ^ iteration)
    random.seed(seed ^ iteration)
    parameter_list, histories = fl.simulation.start_simulation_no_ray(
        client_fn=client_fn,
        num_clients=num_clients,
        client_resources={},
        config=config,
        strategy=strategy,
    )
    save_history(histories, name)
    return parameter_list, histories


def run_sec_agg_demo() -> tuple[list[tuple[int, NDArrays]], History]:
    """Execute the secure aggregation demo."""
    strategy = SecureAggregationStrategy(
        n_dim=PARAMETERS["n_dim"],
        num_clients_per_round=PARAMETERS["num_clients_per_round"],
        threshold=PARAMETERS["min_num_surviving_clients"],
        num_dropouts=PARAMETERS["num_dropouts"],
    )
    return start_seeded_simulation(
        client_fn=get_sec_agg_client_generator(
            PARAMETERS["n_dim"], PARAMETERS["n_samples"], Path(PARAMETERS["data_dir"])
        ),
        num_clients=PARAMETERS["num_total_clients"],
        config=ServerConfig(num_rounds=PARAMETERS["num_rounds"]),
        strategy=strategy,
        name="sec_agg_demo",
    )


if __name__ == "__main__":
    # Create folder for client data
    data_dir = Path(str(PARAMETERS["data_dir"]))
    if not data_dir.exists():
        data_dir.mkdir()
    # Run the demo
    params, hist = run_sec_agg_demo()
    # Clean up caches
    for filename in os.listdir(data_dir):
        file_pth = data_dir / filename
        if filename.endswith(".pth"):
            file_pth.unlink()
