import reedsolo
import random

# Set the number of symbols (nsym) to use for error correction
nsym = 5

# Generate a random 64-bit message to encode
msg = b'1010101010101010'

# Convert the message to a list of integers
msg_int = [int(x) for x in msg]

# Use the rs_encode_msg function to encode the message
encoded = reedsolo.rs_encode_msg(msg_int, nsym)

# Simulate a 16-bit error by flipping 16 random bits in the encoded message
corrupted = encoded.copy()
for i in range(3):
    index = random.randint(0, len(corrupted)-1)
    corrupted[index] = (corrupted[index] + 1) % 256

# Use the rs_correct_msg function to correct errors in the corrupted message
decoded = reedsolo.rs_correct_msg(corrupted, nsym)

# Convert the decoded message back to bytes
decoded_bytes = bytes(list(decoded))

# Print the original message, the corrupted message, and the decoded message
print("Original message:", msg)
print("Corrupted message:", bytes(corrupted))
print("Decoded message:", decoded_bytes)
