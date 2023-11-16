import random
import fieldmath
import reedsolomon


# Runs a bunch of demos and tests, printing information to standard output.
def main():
    show_binary_example()


def subtract_lists(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length for subtraction.")

    result = []
    for i in range(len(list1)):
        result.append(list1[i] - list2[i])

    return result


# Shows an example of encoding a binary message, and decoding a codeword containing errors.
def show_binary_example():
    print("########## Show Binary Example ##########")

    # Configurable parameters
    ''' All possible values for binary field:
        0x1 -- 0x3 -- 0x7 -- 0xB -- 0x13 -- 0x25
        0x43 -- 0x83 -- 0x11D -- 0x211 -- 0x409 -- 0x811 
        
        Values should be irreducible in Z_2 '''
    field = fieldmath.BinaryField(0xB)  # Default: 0x11D=285
    generator = 0x02  # 2
    msglen = 4
    ecclen = 3
    probability = (ecclen // 2) / (msglen + ecclen)  # around half of the length of ecclen errors going to happen
    probability = 1/9

    '''hyperparameters that works properly:
    8 bits -- 0x11D -- 0x02 -- any -- any
    6 bits -- 0x25  -- 0x02 -- 10 -- 10
    
    '''

    # Reed-Solomon instantiate
    rs = reedsolomon.ReedSolomon(field, generator, msglen, ecclen)

    # Generate random message
    message = [random.randrange(field.size) for _ in range(msglen)]
    print(f"Original message: {message}")

    # Encode message to produce codeword
    codeword_org = rs.encode(message)
    codeword_noise = codeword_org.copy()
    print(f"Encoded codeword: {codeword_org}")

    # Perturb some values in the codeword
    perturbed = 0
    for i in range(len(codeword_noise)):
        if random.random() < probability:
            codeword_noise[i] = field.add(codeword_noise[i], random.randrange(1, field.size))
            perturbed += 1
    print(f"Number of values perturbed: {perturbed}")
    print(f"Perturbed codeword: {codeword_noise}")
    print(f"Noisy numbers are in the position of non-zero: {subtract_lists(codeword_org, codeword_noise)}")

    # Try to decode the codeword
    decoded = rs.decode(codeword_noise)
    print(f"Decoded message: {decoded if (decoded is not None) else 'Failure'}")
    print()


if __name__ == "__main__":
    main()
