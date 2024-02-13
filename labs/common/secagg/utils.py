"""Utility function for Secure Aggregation (SA) in Flower.

These functions are for demonstration purposes only.
In practice, one should define and create new message types to facilitate communications
in SA, since adversaries can exploit pickle.loads to conduct attacks.
"""

from enum import IntEnum, auto
import os
import pickle
from dataclasses import dataclass
import base64
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import numpy as np

from Crypto.Util.Padding import pad, unpad
from Crypto.Protocol.SecretSharing import Shamir
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from flwr.common import FitIns, Parameters, ndarrays_to_parameters, Scalar


class SecAggStages(IntEnum):
    """Class for dealing with Secure Aggregation stages."""

    STAGE_0 = 0
    STAGE_1 = auto()
    STAGE_2 = auto()
    STAGE_3 = auto()


def empty_parameters() -> Parameters:
    """Return empty parameters."""
    return ndarrays_to_parameters([])


def save_content(content: Any, d: dict[str, Scalar]) -> dict[str, Scalar]:
    """Save the `content` to the dictionary as a pickle binary and return it."""
    d["content"] = pickle.dumps(content)
    return d


def load_content(d: dict[str, Scalar]) -> Any:
    """Return the pickled `content` of the dictionary and remove it."""
    return pickle.loads(d.pop("content"))


def build_fit_ins(
    content: Any, stage: int, server_round: int, parameters: Parameters | None = None
) -> FitIns:
    """Construct the FitIns message for the server."""
    cfg = save_content(content, {"server_rnd": server_round, "stage": stage})
    return FitIns(
        parameters=parameters if parameters is not None else empty_parameters(),
        config=cfg,
    )


## Key Generation  ====================================================================

def generate_key_pairs() -> (
    tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]
):
    """Generate private and public key pairs with Cryptography."""
    sk = ec.generate_private_key(ec.SECP384R1())
    pk = sk.public_key()
    return sk, pk


def private_key_to_bytes(sk: ec.EllipticCurvePrivateKey) -> bytes:
    """Serialize private key."""
    return sk.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def bytes_to_private_key(b: bytes) -> ec.EllipticCurvePrivateKey:
    """Deserialize private key."""
    return serialization.load_pem_private_key(data=b, password=None)


def public_key_to_bytes(pk: ec.EllipticCurvePublicKey) -> bytes:
    """Serialize public key."""
    return pk.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def bytes_to_public_key(b: bytes) -> ec.EllipticCurvePublicKey:
    """Deserialize public key."""
    return serialization.load_pem_public_key(data=b)


def generate_shared_key(
    sk: ec.EllipticCurvePrivateKey, pk: ec.EllipticCurvePublicKey
) -> bytes:
    """Generate shared key by exchange function and key derivation function.

    Key derivation function is needed to obtain final shared key of exactly 32 bytes
    """
    # Generate a 32 byte urlsafe(for fernet) shared key from own private key and another
    # public key
    sharedk = sk.exchange(ec.ECDH(), pk)
    derivedk = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=None,
    ).derive(sharedk)
    return base64.urlsafe_b64encode(derivedk)


## Authenticated Encryption ============================================================


def encrypt(key: bytes, plaintext: bytes) -> bytes:
    """Encrypt plaintext with Fernet. Key must be 32 bytes."""
    # NOTE: key must be url safe
    f = Fernet(key)
    return f.encrypt(plaintext)


def decrypt(key: bytes, token: bytes) -> bytes:
    """Decrypt ciphertext with Fernet. Key must be 32 bytes."""
    # NOTE: key must be url safe
    f = Fernet(key)
    return f.decrypt(token)


## Random Bytes Generator =============================================================


def rand_bytes(num: int = 32) -> bytes:
    """Generate random bytes with os."""
    return os.urandom(num)


## Arithmetics ========================================================================

def factor_weights_combine(
    weights_factor: int, weights: list[np.ndarray]
) -> list[np.ndarray]:
    """Combine the factor with the weights and return the combined weights."""
    return [np.array([weights_factor])] + weights

def factor_weights_extract(weights: list[np.ndarray]) -> tuple[int, list[np.ndarray]]:
    """Extract the factor from the weights and return the rest of the weights."""
    return weights[0][0], weights[1:]

def weights_shape(weights: list[np.ndarray]) -> list[tuple]:
    """Create a list of shapes of each element in weights."""
    return [arr.shape for arr in weights]

def weights_zero_generate(
    dimensions_list: list[tuple], dtype: type = np.int64
) -> list[np.ndarray]:
    """Generate a list of zero weights based on the dimensions list."""
    return [np.zeros(dimensions, dtype=dtype) for dimensions in dimensions_list]

def weights_addition(a: list[np.ndarray], b: list[np.ndarray]) -> list[np.ndarray]:
    """Add two lists of weights element-wise."""
    return [a[idx] + b[idx] for idx in range(len(a))]

def weights_subtraction(a: list[np.ndarray], b: list[np.ndarray]) -> list[np.ndarray]:
    """Subtract b from a element-wise."""
    return [a[idx] - b[idx] for idx in range(len(a))]

def weights_mod(a: list[np.ndarray], b: int) -> list[np.ndarray]:
    """Take mod of a weights with an integer. If b is a power of 2, use bitwise and."""
    if bin(b).count("1") == 1:
        msk = b - 1
        return [a[idx] & msk for idx in range(len(a))]
    return [a[idx] % b for idx in range(len(a))]

def weights_multiply(a: list[np.ndarray], b: int) -> list[np.ndarray]:
    """Multiply a list of weights by an integer."""
    return [a[idx] * b for idx in range(len(a))]

def weights_divide(a: list[np.ndarray], b: int) -> list[np.ndarray]:
    """Divide a list of weights by an integer."""
    return [a[idx] / b for idx in range(len(a))]


## Quantization ========================================================================

def stochastic_round(arr: np.ndarray) -> np.ndarray[np.int32]:
    """Round stochasticly the input array."""
    ret = np.ceil(arr).astype(np.int32)
    rand_arr = np.random.rand(*ret.shape)
    ret[rand_arr < ret - arr] -= 1
    return ret


def quantize(
    weight: list[np.ndarray], clipping_range: float, target_range: int
) -> list[np.ndarray]:
    """Quantize weight vector to range [-clipping_range, clipping_range]."""
    quantized_list = []
    quantizer = target_range / (2 * clipping_range)
    for arr in weight:
        # Stochastic quantization
        quantized = (
            np.clip(arr, -clipping_range, clipping_range) + clipping_range
        ) * quantizer
        quantized = stochastic_round(quantized)
        quantized_list.append(quantized)
    return quantized_list


def reverse_quantize(
    weight: list[np.ndarray], clipping_range: float, target_range: int
) -> list[np.ndarray]:
    """Transform weight vector to range [-clipping_range, clipping_range]."""
    reverse_quantized_list = []
    quantizer = (2 * clipping_range) / target_range
    shift = -clipping_range
    for arr in weight:
        arr = arr.view(np.ndarray).astype(float) * quantizer + shift
        reverse_quantized_list.append(arr)
    return reverse_quantized_list


## Shamir's secret sharing  ============================================================


def create_shares(secret: bytes, threshold: int, num: int) -> list[bytes]:
    """Create shares with PyCryptodome.

    Each share must be processed to be a byte string with pickle for RPC.
    Return a list of lists for each user. Each sublist contains a share for a 16 byte
    chunk of the secret. The integer part of the tuple represents the index of the
    share, not the index of the chunk it is representing.
    """
    secret_padded = pad(secret, 16)
    secret_padded_chunk = [
        (threshold, num, secret_padded[i : i + 16])
        for i in range(0, len(secret_padded), 16)
    ]
    share_list = []
    for _ in range(num):
        share_list.append([])

    with ThreadPoolExecutor(max_workers=10) as executor:
        for chunk_shares in executor.map(
            lambda arg: shamir_split(*arg), secret_padded_chunk
        ):
            for idx, share in chunk_shares:
                # NOTE: `idx`` start with 1
                share_list[idx - 1].append((idx, share))

    for idx, shares in enumerate(share_list):
        share_list[idx] = pickle.dumps(shares)

    return share_list


def shamir_split(threshold: int, num: int, chunk: bytes) -> list[tuple[int, bytes]]:
    """Call the Shamir split function on the `threshold`, `num`, and `chunk`."""
    return Shamir.split(threshold, num, chunk)


def combine_shares(share_list: list[bytes]) -> bytes:
    """Reconstruct secret with PyCryptodome combining `shares`."""
    for idx, share in enumerate(share_list):
        share_list[idx] = pickle.loads(share)

    chunk_num = len(share_list[0])
    secret_padded = bytearray(0)
    chunk_shares_list = []
    for i in range(chunk_num):
        chunk_shares = []
        for j in range(len(share_list)):
            chunk_shares.append(share_list[j][i])
        chunk_shares_list.append(chunk_shares)

    with ThreadPoolExecutor(max_workers=10) as executor:
        for chunk in executor.map(shamir_combine, chunk_shares_list):
            secret_padded += chunk

    secret = unpad(secret_padded, 16)
    return bytes(secret)


def shamir_combine(shares: list[tuple[int, bytes]]) -> bytes:
    """Call the Shamir combine function of the `shares`."""
    return Shamir.combine(shares)


## Miscrellaneous ======================================================================


def share_keys_plaintext_concat(
    source: int, destination: int, b_share: bytes, sk_share: bytes
) -> bytes:
    """Concat the attributes of a `ShareKeysPacket` into bytes.

    Unambiguous string concatenation of source, destination, and two secret shares.
    We assume they do not contain the 'abcdef' string.
    """
    source, destination = int.to_bytes(source, 4, "little"), int.to_bytes(
        destination, 4, "little"
    )
    return b"".join(
        [
            source,
            destination,
            int.to_bytes(len(b_share), 4, "little"),
            b_share,
            sk_share,
        ]
    )


def share_keys_plaintext_separate(plaintext: bytes) -> tuple[int, int, bytes, bytes]:
    """Interpret the plain text into the attributes of a `ShareKeysPacket`.

    Unambiguous string splitting to obtain source, destination and two secret shares
    """
    src, dst, mark = (
        int.from_bytes(plaintext[:4], "little"),
        int.from_bytes(plaintext[4:8], "little"),
        int.from_bytes(plaintext[8:12], "little"),
    )
    ret = [src, dst, plaintext[12 : 12 + mark], plaintext[12 + mark :]]
    return ret


# Pseudo Bytes Generator ===============================================================

def pseudo_rand_gen(
    seed: bytes, num_range: int, dimensions_list: list[tuple]
) -> list[np.ndarray]:
    """Implement a pseudo random generator for creating masks."""
    assert len(seed) & 0x3 == 0
    seed32 = 0
    for i in range(0, len(seed), 4):
        seed32 ^= int.from_bytes(seed[i : i + 4], "little")
    np.random.seed(seed32)
    output = []
    for dimension in dimensions_list:
        if len(dimension) == 0:
            arr = np.array(np.random.randint(0, num_range - 1), dtype=int)
        else:
            arr = np.random.randint(0, num_range - 1, dimension)
        output.append(arr)
    return output


@dataclass
class ShareKeysPacket:
    """Dataclass for packets of shared keys."""

    source: int
    destination: int
    ciphertext: bytes
