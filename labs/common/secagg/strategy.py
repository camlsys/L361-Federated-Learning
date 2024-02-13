"""Strategy for Secure Aggregation."""

from logging import INFO
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import (
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    log,
)
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import numpy as np

from common.secagg.utils import (
    SecAggStages,
    ShareKeysPacket,
    bytes_to_private_key,
    bytes_to_public_key,
    combine_shares,
    empty_parameters,
    factor_weights_extract,
    generate_shared_key,
    load_content,
    pseudo_rand_gen,
    reverse_quantize,
    save_content,
    weights_addition,
    weights_divide,
    weights_mod,
    weights_shape,
    weights_subtraction,
)


from flwr.common import (
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)


class SecureAggregationStrategy(FedAvg):
    """Flower strategy for secure aggregation."""

    def __init__(
        self,
        *,
        n_dim: int,
        num_clients_per_round: int,
        threshold: float,
        num_dropouts: int,
               fraction_fit: float = 0.0,
        fraction_evaluate: float = 0.0,
        min_fit_clients: int = 0,
        min_evaluate_clients: int = 0,
        min_available_clients: int = 0,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=num_clients_per_round,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=num_clients_per_round,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        
        self.n_dim = n_dim
        self.sample_num = num_clients_per_round
        self.threshold = threshold
        self.dropout_num = num_dropouts

        # Runtime variables
        self.proxy2id = {}
        self.stage = 0
        self.surviving_clients = {}
        self.public_keys_dict = {}
        self.forward_packet_list_dict = {}
        self.masked_vector = []
        self.dropout_clients = {}

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        """Initialize the (global) model parameters."""
        return empty_parameters()

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {"server_rnd": server_round, "stage": self.stage}
        tmp_ret: list[tuple[ClientProxy, FitIns]] = []
        log(logging.INFO, f"Configure fit: stage {self.stage}, {SecAggStages.STAGE_0}, {self.stage == SecAggStages.STAGE_0}")
        if self.stage == SecAggStages.STAGE_0:
            log(logging.INFO, "Stage 0")
            config["share_num"] = self.sample_num
            config["threshold"] = self.threshold
            self.proxy2id = {}
            self.surviving_clients = {}
            # Sample clients
            clients = client_manager.sample(
                num_clients=self.sample_num, min_num_clients=self.sample_num
            )
            for idx, client in enumerate(clients):
                self.proxy2id[client] = idx
                cfg = config.copy()
                cfg["id"] = idx
                cfg["drop_flag"] = idx < self.dropout_num
                tmp_ret += [(client, FitIns(empty_parameters(), cfg))]
        if self.stage == SecAggStages.STAGE_1:
            save_content(self.public_keys_dict, config)
            fit_ins = FitIns(empty_parameters(), config)
            tmp_ret = [(client, fit_ins) for client in self.surviving_clients.values()]
        if self.stage == SecAggStages.STAGE_2:
            # Fit Instructions here
            fit_ins_lst = [FitIns(empty_parameters(), {})] * self.sample_num

            for idx, client in self.surviving_clients.items():
                assert idx == self.proxy2id[client]
                tmp_ret.append(
                    (
                        client,
                        FitIns(
                            empty_parameters(),
                            save_content(
                                (self.forward_packet_list_dict[idx], fit_ins_lst[idx]),
                                config.copy(),
                            ),
                        ),
                    )
                )
        if self.stage == SecAggStages.STAGE_3:
            save_content(
                (
                    list(self.surviving_clients.keys()),
                    list(self.dropout_clients.keys()),
                ),
                config,
            )
            fit_ins = FitIns(empty_parameters(), config)
            tmp_ret = [(client, fit_ins) for client in self.surviving_clients.values()]
        return tmp_ret

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate training results."""
        if not results and failures:
            for f in failures:
                if isinstance(f, BaseException):
                    raise f
        for f in failures:
            log(INFO, f)
        log(INFO, "Length of content is %s", len(results[0][1].metrics["content"]))
        parameters_cost = sum(
            [sum(len(t) for t in r[1].parameters.tensors) for r in results]
        )
        content_cost = sum([len(r[1].metrics["content"]) for r in results])
        if self.stage == SecAggStages.STAGE_0:
            public_keys_dict: dict[int, tuple[bytes, bytes]] = {}
            ask_keys_results = results
            if len(ask_keys_results) < self.threshold:
                raise Exception(  # noqa: TRY002
                    "Not enough available clients after ask keys stage"
                )
            share_keys_clients: dict[int, ClientProxy] = {}

            # Build public keys dict
            # tmp_map = dict(ask_keys_results)
            for client, result in ask_keys_results:
                idx = self.proxy2id[client]
                public_keys_dict[idx] = load_content(result.metrics)
                share_keys_clients[idx] = client
            self.public_keys_dict = public_keys_dict
            self.surviving_clients = share_keys_clients
        elif self.stage == SecAggStages.STAGE_1:
            # Build forward packet list dictionary
            total_packet_list: list[ShareKeysPacket] = []
            forward_packet_list_dict: dict[int, list[ShareKeysPacket]] = {}
            ask_vectors_clients: dict[int, ClientProxy] = {}
            for client, fit_res in results:
                idx = self.proxy2id[client]
                result = load_content(fit_res.metrics)
                ask_vectors_clients[idx] = client
                packet_list = result
                total_packet_list += packet_list

            for idx in ask_vectors_clients.keys():
                forward_packet_list_dict[idx] = []
            for packet in total_packet_list:
                destination = packet.destination
                if destination in ask_vectors_clients.keys():
                    forward_packet_list_dict[destination].append(packet)
            self.surviving_clients = ask_vectors_clients
            self.forward_packet_list_dict = forward_packet_list_dict
        elif self.stage == SecAggStages.STAGE_2:
            if len(results) < self.threshold:
                raise Exception(  # noqa: TRY002
                    "Not enough available clients after ask vectors stage"
                )
            # Get shape of vector sent by first client
            masked_vector = [np.array([0], dtype=int), np.zeros(self.n_dim, dtype=int)]
            # Add all collected masked vectors and compuute available and dropout clients set
            unmask_vectors_clients: dict[int, ClientProxy] = {}
            dropout_clients = self.surviving_clients.copy()
            for client, fit_res in results:
                idx = self.proxy2id[client]
                unmask_vectors_clients[idx] = client
                dropout_clients.pop(idx)
                client_parameters = fit_res.parameters
                masked_vector = weights_addition(
                    masked_vector, parameters_to_ndarrays(client_parameters)
                )

            masked_vector = weights_mod(masked_vector, 1 << 24)
            self.masked_vector = masked_vector
            self.surviving_clients = unmask_vectors_clients
            self.dropout_clients = dropout_clients
        elif self.stage == SecAggStages.STAGE_3:
            # Build collected shares dict
            collected_shares_dict: dict[int, list[bytes]] = {}
            for idx in self.proxy2id.values():
                collected_shares_dict[idx] = []

            if len(results) < self.threshold:
                raise Exception(  # noqa: TRY002
                    "Not enough available clients after unmask vectors stage"
                )
            for _, fit_res in results:
                share_dict = load_content(fit_res.metrics)
                for owner_id, share in share_dict.items():
                    collected_shares_dict[owner_id].append(share)
            masked_vector = self.masked_vector
            # Remove mask for every client who is available before ask vectors stage,
            # Divide vector by first element
            for client_id, share_list in collected_shares_dict.items():
                if len(share_list) < self.threshold:
                    raise Exception(  # noqa: TRY002
                        "Not enough shares to recover secret in unmask vectors stage"
                    )
                secret = combine_shares(share_list)
                if client_id in self.surviving_clients.keys():
                    # seed is an available client's b
                    private_mask = pseudo_rand_gen(
                        secret, 1 << 24, weights_shape(masked_vector)
                    )
                    masked_vector = weights_subtraction(masked_vector, private_mask)
                else:
                    # seed is a dropout client's sk1
                    neighbor_list = list(self.proxy2id.values())
                    neighbor_list.remove(client_id)

                    for neighbor_id in neighbor_list:
                        shared_key = generate_shared_key(
                            bytes_to_private_key(secret),
                            bytes_to_public_key(self.public_keys_dict[neighbor_id][0]),
                        )
                        pairwise_mask = pseudo_rand_gen(
                            shared_key, 1 << 24, weights_shape(masked_vector)
                        )
                        if client_id > neighbor_id:
                            masked_vector = weights_addition(
                                masked_vector, pairwise_mask
                            )
                        else:
                            masked_vector = weights_subtraction(
                                masked_vector, pairwise_mask
                            )
            masked_vector = weights_mod(masked_vector, 1 << 24)
            # Divide vector by number of clients who have given us their masked vector
            # i.e. those participating in final unmask vectors stage
            total_weights_factor, masked_vector = factor_weights_extract(masked_vector)
            masked_vector = weights_divide(masked_vector, total_weights_factor)
            aggregated_vector = reverse_quantize(masked_vector, 3, 1 << 16)
            # log(INFO, aggregated_vector[:4])
            aggregated_parameters = ndarrays_to_parameters(aggregated_vector)

            self.stage = 0
            return aggregated_parameters, {"total_cost": parameters_cost + content_cost}

        self.stage = (self.stage + 1) % 4

        return None, {"total_cost": parameters_cost + content_cost}
