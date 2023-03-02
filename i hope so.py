import reedsolo
import numpy as np
import torch

def rs_encode(msg, ecc_bytes):
    """
    Encode a binary message using Reed-Solomon error-correcting code.

    Parameters:
        msg (torch.Tensor): A 1D tensor of binary numbers (0 or 1) representing the message to be encoded.
        ecc_bytes (int): The number of error-correction bytes to add to the message.

    Returns:
        torch.Tensor: A 1D tensor of binary numbers (0 or 1) representing the encoded message.
    """
    # Convert the binary message to a numpy array of bytes
    msg_np = np.packbits(msg.numpy())

    # Use reedsolo to encode the message
    ecc_np = reedsolo.rs_encode_msg(msg_np, ecc_bytes)

    # Convert the encoded message back to a torch tensor of binary numbers
    ecc = torch.from_numpy(np.unpackbits(ecc_np))

    return ecc


def rs_decode(ecc, ecc_bytes):
    """
    Decode a binary message that has been encoded using Reed-Solomon error-correcting code.

    Parameters:
        ecc (torch.Tensor): A 1D tensor of binary numbers (0 or 1) representing the message to be decoded, including error-correction bytes.
        ecc_bytes (int): The number of error-correction bytes that were added to the original message.

    Returns:
        torch.Tensor: A 1D tensor of binary numbers (0 or 1) representing the decoded message.
    """
    # Convert the binary message to a numpy array of bytes
    ecc_np = np.packbits(ecc.numpy())

    # Use reedsolo to decode the message
    msg_np = reedsolo.rs_correct_msg(ecc_np, ecc_bytes)

    # Convert the decoded message back to a torch tensor of binary numbers
    msg = torch.from_numpy(np.unpackbits(msg_np))

    return msg


# Example usage
msg = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0])
print("Original Message:", msg)
P = 2 # 2 parity bits for error correction

# Encode the message
ecc = rs_encode(msg, P)
print("Encoded message:", ecc)

# Simulate some errors (flip some bits)
error_indices = [1, 3, 5]
ecc[error_indices] = 1 - ecc[error_indices]
print("Received message with errors:", ecc)

# Decode the message
decoded_msg = rs_decode(ecc, P)
print("Decoded message:", decoded_msg)
