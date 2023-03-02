import random
import zlib

# Generate a random 64-bit binary message
message = ''.join(str(random.randint(0, 1)) for i in range(64)).encode('utf-8')

# Introduce random errors in the message (20% of bits flipped)
error_bits = random.sample(range(64), k=64//5)
message = bytearray(message)
for i in error_bits:
    message[i//8] ^= (1 << (i%8))

# Add a CRC-32 checksum to the message
checksum = zlib.crc32(message)
print(checksum)
# Verify the checksum and recover the original message
if zlib.crc32(message) == checksum:
    print("Message received correctly!")
    original_message = message
else:
    print("Error in message transmission, message corrupted.")
    # Try to recover the original message
    for i in error_bits:
        message[i//8] ^= (1 << (i%8))
    if zlib.crc32(message) == checksum:
        print("Original message successfully recovered!")
        original_message = message
    else:
        print("Original message could not be recovered.")



########################################



import random
import zlib

# Generate a random 64-bit binary message
message = ''.join(str(random.randint(0, 1)) for i in range(64))

# Introduce random errors in the message (20% of bits flipped)
error_bits = random.sample(range(64), k=64//5)
message = list(message)
for i in error_bits:
    message[i] = str(int(message[i]) ^ 1)
message = ''.join(message)

# Add a CRC-32 checksum to the message
checksum = format(zlib.crc32(message.encode('utf-8')), '032b')

# Verify the checksum and recover the original message
if format(zlib.crc32(message.encode('utf-8')), '032b') == checksum:
    print("Message received correctly!")
    original_message = message
else:
    print("Error in message transmission, message corrupted.")
    # Try to recover the original message
    message = list(message)
    for i in error_bits:
        message[i] = str(int(message[i]) ^ 1)
    message = ''.join(message)
    if format(zlib.crc32(message.encode('utf-8')), '032b') == checksum:
        print("Original message successfully recovered!")
        original_message = message
    else:
        print("Original message could not be recovered.")
