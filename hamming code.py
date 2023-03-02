import bitarray

def generate_hamming_code(data_bits):
    # Calculate the number of parity bits needed
    r = 1
    while 2**r < len(data_bits) + r + 1:
        r += 1

    # Initialize the encoded message with zeros
    encoded = bitarray.bitarray('0') * (len(data_bits) + r)

    # Copy the data bits into the encoded message
    j = 0
    for i in range(len(encoded)):
        if i+1 == 2**j:
            j += 1
        else:
            encoded[i] = data_bits[i-j]

    # Calculate the parity bits
    for i in range(r):
        pos = 2**i - 1
        xor = False
        for j in range(pos, len(encoded), 2*pos+2):
            xor ^= encoded[j:j+pos+1].count(True) % 2 == 1
        encoded[pos] = int(xor)

    return encoded

def hamming_decode(encoded_bits):
    # Determine the number of parity bits used in the encoded message
    r = 1
    while 2**r < len(encoded_bits):
        r += 1

    # Initialize the decoded message with zeros
    decoded = bitarray.bitarray('0') * (len(encoded_bits) - r)

    # Calculate the parity bits and syndrome
    syndrome = 0
    for i in range(r):
        pos = 2**i - 1
        xor = False
        for j in range(pos, len(encoded_bits), 2*pos+2):
            xor ^= encoded_bits[j:j+pos+1].count(True) % 2 == 1
        syndrome |= int(xor) << i

    # Correct the errors, if any
    if syndrome != 0:
        decoded[syndrome-1] = not decoded[syndrome-1]

    # Copy the data bits from the encoded message into the decoded message
    j = 0
    for i in range(len(encoded_bits)):
        if i+1 != 2**j:
            decoded[i-j] = encoded_bits[i]
        else:
            j += 1

    return decoded



# Convert a string of bits to a bitarray
original = bitarray.bitarray('1101011100011111')
print("Original message: ", original.to01())

# Encode the message
encoded = generate_hamming_code(original)
print("Encoded message: ", encoded.to01())
# Introduce errors into the encoded message
for i in range(len(encoded)):
    if i % 5 == 0:
        encoded[i] = not encoded[i]
print("Encoded message: ", encoded.to01())

# Decode the message
decoded = hamming_decode(encoded)
print("Decoded message: ", decoded.to01())

# Print the results
print("Percent accuracy: ", original.count() / len(original) * 100)
