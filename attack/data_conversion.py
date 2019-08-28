import torch
from models.quantization import quan_Conv2d, quan_Linear


def int2bin(input, num_bits):
    '''
    convert the signed integer value into unsigned integer (2's complement equivalently).
    '''
    output = input.clone()
    output[input.lt(0)] = 2**num_bits + output[input.lt(0)]
    return output


def bin2int(input, num_bits):
    '''
    convert the unsigned integer (2's complement equivantly) back to the signed integer format
    with the bitwise operations. Note that, in order to perform the bitwise operation, the input
    tensor has to be in the integer format.
    '''
    mask = 2**(num_bits - 1) - 1
    output = -(input & ~mask) + (input & mask)
    return output


def weight_conversion(model):
    '''
    Perform the weight data type conversion between:
        signed integer <==> two's complement (unsigned integer)

    Note that, the data type conversion chosen is depend on the bits:
        N_bits <= 8   .char()   --> torch.CharTensor(), 8-bit signed integer
        N_bits <= 16  .short()  --> torch.shortTensor(), 16 bit signed integer
        N_bits <= 32  .int()    --> torch.IntTensor(), 32 bit signed integer
    '''
    for m in model.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            w_bin = int2bin(m.weight.data, m.N_bits).short()
            m.weight.data = bin2int(w_bin, m.N_bits).float()
    return


def count_ones(t, n_bits):
    counter = 0
    for i in range(n_bits):
        counter += ((t & 2**i) // 2**i).sum()
    return counter.item()


def hamming_distance(model1, model2):
    '''
    Given two model whose structure, name and so on are identical.
    The only difference between the model1 and model2 are the weight.
    The function compute the hamming distance bewtween the bianry weights
    (two's complement) of model1 and model2.
    '''
    # TODO: add the function check model1 and model2 are same structure
    # check the keys of state_dict match or not.

    H_dist = 0  # hamming distance counter

    for name, module in model1.named_modules():
        if isinstance(module, quan_Conv2d) or isinstance(module, quan_Linear):
            # remember to convert the tensor into integer for bitwise operations
            binW_model1 = int2bin(model1.state_dict()[name + '.weight'],
                                  module.N_bits).short()
            binW_model2 = int2bin(model2.state_dict()[name + '.weight'],
                                  module.N_bits).short()
            H_dist += count_ones(binW_model1 ^ binW_model2, module.N_bits)

    return H_dist