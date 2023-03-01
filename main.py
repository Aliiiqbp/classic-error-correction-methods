from reedsolo import RSCodec
import numpy as np

# Generate a random 144-bit binary message
msg = "".join([str(np.random.randint(2)) for i in range(144)])
print("Original message:", msg)

# Number of symbols for error correction
nsym = 15
rs = RSCodec()

# Add redundancy to the message using the Reed-Solomon code
encoded_msg = rs.encode(bytes(msg, encoding='utf8'), nsym)
print("Encoded message:", encoded_msg)

# Introduce errors into the message (simulate transmission errors)
error_rate = 0.5
num_errors = int(error_rate * len(encoded_msg))
err_pos = np.random.choice(len(encoded_msg), num_errors, replace=False)
corrupted_msg = "".join(["0" if i in err_pos else encoded_msg[i] for i in range(len(encoded_msg))])
print("Corrupted message:", corrupted_msg)


# Correct the errors in the message using the Reed-Solomon code
decoded_msg = rs.decode(corrupted_msg, nsym)
print("Decoded message:", decoded_msg)


# Verify that the decoded message matches the original message
assert decoded_msg == msg

print("Encoded message:", encoded_msg)
print("Corrupted message:", corrupted_msg)
print("Decoded message:", decoded_msg)
