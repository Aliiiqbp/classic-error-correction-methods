import torch
import numpy as np
import reedsolo

def binary_encode(msg):
    return torch.ByteTensor(list(map(int, list(msg))))

def binary_decode(msg):
    return ''.join(map(str, list(msg.numpy())))

def rs_encode(msg: torch.Tensor, nsym: int) -> torch.Tensor:
    # convert message tensor to numpy array
    msg_np = msg.cpu().numpy()
    # encode message using Reed-Solomon code
    ecc = reedsolo.rs_encode_msg(msg_np, nsym)
    # convert encoded message to PyTorch tensor
    ecc = torch.from_numpy(np.frombuffer(ecc, dtype=np.uint8))
    return ecc
def rs_decode(codeword: torch.Tensor, nsym: int) -> torch.Tensor:
    codeword_np = codeword.numpy()
    msg_out = reedsolo.rs_correct_msg(ecc, nsym, len(ecc))
    return torch.from_numpy(msg_out)

# Example usage
torch.manual_seed(42)
msg = torch.randint(low=0, high=2, size=(144,), dtype=torch.uint8)
nsym = 72
ecc = rs_encode(msg, nsym)

# Simulate errors by flipping some bits
corrupted = ecc.clone()
idxs = torch.randperm(len(corrupted))[:16]
corrupted[idxs] = 1 - corrupted[idxs]

# Decode the corrupted message
decoded = rs_decode(corrupted, nsym)

# Print the results
print("Original Message: ", binary_decode(msg))
print("Encoded Message: ", binary_decode(ecc))
print("Corrupted Message: ", binary_decode(corrupted))
print("Decoded Message: ", binary_decode(decoded))
print("Errors Corrected: ", (decoded != msg).sum().item())
