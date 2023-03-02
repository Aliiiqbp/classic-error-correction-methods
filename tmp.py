import reedsolo

# Define the message and number of symbols for error correction
msg = b'\x12\x34\x56\x78\x9a\xbc\xde\xf0'
nsym = 5

# Initialize the Reed-Solomon coder and encode the message
rs = reedsolo.RSCodec(nsym)
encoded = rs.encode(msg)

# Simulate errors by flipping some of the bits in the encoded message
corrupted = bytearray(encoded)
corrupted[3] ^= 0b00001000
corrupted[7] ^= 0b00100000

# Decode the corrupted message and correct errors
decoded = rs.decode(corrupted)

# Print the original message, encoded message, corrupted message, and decoded message
print(f"Original message: {msg}")
print(f"Encoded message: {encoded}")
print(f"Corrupted message: {corrupted}")
print(f"Decoded message: {decoded}")





import reedsolo

# Define the message and number of symbols for error correction
msg = b'\x12\x34\x56\x78\x9a\xbc\xde\xf0'
nsym = 6

# Initialize the Reed-Solomon coder and encode the message
rs = reedsolo.RSCodec(nsym)
encoded = rs.encode(msg)

# Simulate errors by flipping some of the bits in the encoded message
corrupted = bytearray(encoded)
corrupted[5] ^= 0b00100000  # flip bit 5
corrupted[9] ^= 0b00000010  # flip bit 9
corrupted[11] ^= 0b01000000  # flip bit 11

# Decode the corrupted message and correct errors
decoded = rs.decode(corrupted)

# Print the original message, encoded message, corrupted message, and decoded message
print(f"Original message: {msg}")
print(f"Encoded message: {encoded}")
print(f"Corrupted message: {corrupted}")
print(f"Decoded message: {decoded}")
