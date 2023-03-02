import torch
import reedsolo
from math import ceil
import numpy as np


def rs_encode(msg: torch.Tensor, P: float) -> torch.Tensor:
    n = msg.numel()
    nsym = ceil(n * P)
    msg_np = msg.numpy().astype(np.uint8)
    ecc_np = reedsolo.rs_encode_msg(msg_np, nsym)
    ecc = torch.from_numpy(np.asarray(ecc_np))
    return ecc


# example usage
msg = torch.randint(0, 2, (36,))
print(msg, len(msg))
encoded = rs_encode(msg, 0.4)
print(encoded, len(encoded))
