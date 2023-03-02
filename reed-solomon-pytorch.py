import torch


def gf2_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Add two polynomials in GF(2)
    """
    res = torch.zeros(max(a.shape[0], b.shape[0]), dtype=torch.uint8)
    res[-a.shape[0]:] = a
    res[-b.shape[0]:] ^= b
    return res


def gf2_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two polynomials in GF(2)
    """
    res = torch.zeros(a.shape[0] + b.shape[0] - 1, dtype=torch.uint8)
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            res[i+j] ^= a[i] & b[j]
    return res


def gf2_div(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Divide two polynomials in GF(2)
    """
    a_deg = gf2_deg(a)
    b_deg = gf2_deg(b)
    q = torch.zeros(max(a_deg - b_deg + 1, 0), dtype=torch.uint8)
    r = a.clone()
    for i in range(a_deg - b_deg + 1):
        coef = r[a_deg - i] // b[b_deg]
        q[-i-1] = coef
        r[a_deg - i - b_deg:a_deg - i + 1] ^= gf2_mul(b[::-1], torch.tensor([coef], dtype=torch.uint8))
    return q, r


def gf2_deg(a: torch.Tensor) -> int:
    """
    Calculate degree of a polynomial in GF(2)
    """
    for i in range(a.shape[0]-1, -1, -1):
        if a[i] != 0:
            return i
    return -1


def rs_encode(msg: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """
    Encode a message with Reed-Solomon
    """
    gen_poly = torch.tensor([1], dtype=torch.uint8)
    for i in range(n - k):
        gen_poly = gf2_mul(gen_poly, torch.tensor([1, 2**i], dtype=torch.uint8))
    msg_padded = torch.zeros(n - k + len(msg), dtype=torch.uint8)
    msg_padded[-len(msg):] = msg
    res = torch.zeros(n, dtype=torch.uint8)
    for i in range(k):
        coef = msg_padded[i]
        if coef != 0:
            res[i:i+len(gen_poly)] ^= gf2_mul(coef, gen_poly)
    res[k:] = msg_padded[k:]
    return res


def rs_decode(codeword: torch.Tensor, n: int, k: int) -> torch.Tensor:
    syndrome = torch.zeros(n-k, dtype=torch.uint8)
    gen_poly = torch.tensor([1], dtype=torch.uint8)
    for i in range(n - k):
        gen_poly = gf2_mul(gen_poly, torch.tensor([1, 2**i], dtype=torch.uint8))
    r = len(gen_poly) - 1
    for i in range(n - k):
        coef = codeword[i]
        if coef != 0:
            syndrome[-i-1:] = gf2_add(syndrome[-i-1:], gf2_mul(coef, gen_poly[::-1]))
    errors = torch.zeros(n, dtype=torch.uint8)
    for i, s in enumerate(syndrome):
        if s != 0:
            errors[n-k-i-1] = 1
    if errors.sum() == 0:
        return codeword[:k]
    locator_poly = torch.tensor([1], dtype=torch.uint8)
    for i in range(n):
        if errors[n-i-1] != 0:
            locator_poly = gf2_mul(locator_poly, torch.tensor([1, 2**i], dtype=torch.uint8))
    for i in range(k):
        coef = codeword[i]
        if coef != 0:
            locator_poly[i:i+r] = gf2_add(locator_poly[i:i+r], gf2_mul(coef, gen_poly))
    roots = []
    for i in range(2**n):
        if gf2_mul(locator_poly, torch.tensor([1, i], dtype=torch.uint8)).sum() == 0:
            roots.append(i)
    if len(roots) != r:
        return None
    for i, root in enumerate(roots):
        x = torch.tensor([root], dtype=torch.uint8)
        y = codeword[k-1-i]
        if y != 0:
            codeword[:k] = gf2_add(codeword[:k], gf2_mul(y, gf2_div(gf2_mul(locator_poly, x), gen_poly)))
    return codeword[:k]





def rs_encode_binary(msg: torch.Tensor, n: int, k: int) -> torch.Tensor:
    # Reshape msg to a 2D tensor with shape (num_blocks, block_size)
    block_size = n - k
    num_blocks = (msg.shape[0] + block_size - 1) // block_size
    padded_msg = torch.cat([msg, torch.zeros(num_blocks * block_size - msg.shape[0], dtype=torch.uint8)], dim=0)
    msg_blocks = padded_msg.reshape(num_blocks, block_size)

    # Generate generator polynomial
    gen_poly = torch.tensor([1], dtype=torch.uint8)
    for i in range(k):
        gen_poly = gf2_mul(gen_poly, torch.tensor([1, gf2_exp(i)], dtype=torch.uint8))

    # Compute remainder
    codewords = []
    for i in range(num_blocks):
        msg_block = msg_blocks[i]
        msg_poly = torch.cat([msg_block, torch.zeros(k - 1, dtype=torch.uint8)], dim=0)
        remainder = gf2_poly_div(msg_poly, gen_poly)[1]
        codeword = torch.cat([msg_block, remainder], dim=0)
        codewords.append(codeword)

    # Concatenate codewords into a single tensor
    return torch.cat(codewords, dim=0)


def rs_decode_binary(codeword: torch.Tensor, n: int, k: int) -> torch.Tensor:
    pad_size = len(codeword) % n
    if pad_size != 0:
        codeword = codeword[:-pad_size]
    chunks = torch.split(codeword, n, dim=1)
    decoded_chunks = []
    for chunk in chunks:
        msg = chunk[:, :k]
        parity = chunk[:, k:]
        decoded_chunk = rs_decode(chunk, n, k)
        if decoded_chunk is None:
            return None
        decoded_chunk = decoded_chunk.unsqueeze(0)
        decoded_chunks.append(decoded_chunk)
    return torch.cat(decoded_chunks, dim=1)















"""example"""

import numpy as np

def generate_random_binary_array(length: int, p: float) -> np.ndarray:
    return np.random.choice([0, 1], size=length, p=[p, 1-p])

# Generate random message
msg = torch.tensor(generate_random_binary_array(144, 0.5), dtype=torch.uint8)

# Specify Reed-Solomon code parameters
n = 15
k = 9

# Encode the message
encoded_msg = rs_encode_binary(msg, n, k)

# Introduce random errors into the encoded message
corrupted_msg = encoded_msg.clone()
errors = torch.tensor(generate_random_binary_array(len(encoded_msg), 0.5), dtype=torch.uint8)
corrupted_msg = torch.bitwise_xor(corrupted_msg, errors)

# Decode the corrupted message
recovered_msg = rs_decode_binary(corrupted_msg, n, k)

# Print the original message, encoded message, corrupted message, and recovered message
print("Original message:", msg)
print("Encoded message:", encoded_msg)
print("Corrupted message:", corrupted_msg)
print("Recovered message:", recovered_msg)


