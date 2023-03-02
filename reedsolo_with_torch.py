import reedsolo
import random

# Set the number of symbols (nsym) to use for error correction
nsym = 4

# Generate a random 64-bit message to encode
# msg = b'1001010f10101110'

msg = [random.randint(0, 15) for _ in range(144)]
msg_bytes = bytearray(msg)


# Convert the message to a list of integers
msg_int = [int(x) for x in msg]

# Use the rs_encode_msg function to encode the message
encoded = reedsolo.rs_encode_msg(msg_int, nsym)

# Convert the encoded message back to bytes
encoded_bytes = bytes(encoded)

# Print the encoded message
print(encoded_bytes)
print("Random message:", ''.join(str(x) for x in encoded_bytes))
# print(bin(int(msg, 16))[2:].zfill(8))


# Generate a random 144-bit message
# msg = [random.randint(0, 1) for _ in range(144)]

# Convert the message to a bytearray
# msg_bytes = bytearray(msg)
# print(msg_bytes)
# Print the message as a binary string
# print("Random message:", ''.join(str(x) for x in msg))
