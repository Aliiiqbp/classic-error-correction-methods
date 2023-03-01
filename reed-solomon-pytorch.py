import torch


def gf2_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Adds two binary numbers (GF(2))"""
    return torch.logical_xor(a, b).to(torch.uint8)


def gf2_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiplies two binary numbers (GF(2))"""
    res = torch.zeros(len(a) + len(b) - 1, dtype=torch.uint8)
    for i in range(len(a)):
        if a[i] != 0:
            res[i:i+len(b)] = gf2_add(res[i:i+len(b)], torch.logical_and(b, a[i]))
    return res


def gf2_div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Divides two binary numbers (GF(2))"""
    deg_a = gf2_deg(a)
    deg_b = gf2_deg(b)
    if deg_b == -1:
        raise ZeroDivisionError("division by zero")
    if deg_a < deg_b:
        return torch.zeros(1, dtype=torch.uint8)
    q = torch.zeros(deg_a - deg_b + 1, dtype=torch.uint8)
    while deg_a >= deg_b:
        d = torch.zeros(deg_a - deg_b + 1, dtype=torch.uint8)
        d[deg_a - deg_b] = 1
        q = gf2_add(q, d)
        a = gf2_add(a, torch.cat([torch.zeros(deg_a - deg_b + 1, dtype=torch.uint8), gf2_mul(d, b[deg_b:])]))
        deg_a = gf2_deg(a)
    return q


def gf2_deg(poly: torch.Tensor) -> int:
    """Returns the degree of a binary polynomial over GF(2)"""
    deg = len(poly) - 1
    while deg >= 0 and poly[deg] == 0:
        deg -= 1
    return deg


def rs_encode(msg: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Encodes a message using Reed-Solomon (binary)"""
    gen_poly = torch.tensor([1], dtype=torch.uint8)
    for i in range(k):
        gen_poly = gf2_mul(gen_poly, torch.tensor([1, 2**i], dtype=torch.uint8))
    codeword = torch.cat([msg, torch.zeros(n-k, dtype=torch.uint8)])
    for i in range(len(msg)):
        coef = codeword[i]
        if coef != 0:
            codeword[i:i+len(gen_poly)] = gf2_add(codeword[i:i+len(gen_poly)], gf2_mul(coef, gen_poly))
    return codeword


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
            codeword[:k] = gf2_add(codeword[:k], gf2_mul(y, gf2_div(locator_poly, gf2_add(x, gen_poly))))
    return codeword[:k]





def rs_encode_binary(msg: torch.Tensor, n: int, k: int) -> torch.Tensor:
    msg = msg.unsqueeze(0)
    pad_size = n - k - (len(msg) % (n - k))
    padded_msg = torch.cat([msg, torch.zeros(pad_size, dtype=torch.uint8)], dim=1)
    chunks = torch.split(padded_msg, k, dim=1)
    encoded_chunks = []
    for chunk in chunks:
        encoded_chunk = torch.cat([chunk, rs_encode(chunk, n, k)], dim=1)
        encoded_chunks.append(encoded_chunk)
    return torch.cat(encoded_chunks, dim=1)


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





