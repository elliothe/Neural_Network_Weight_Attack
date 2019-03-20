import torch
from models.quantization import quan_Conv2d, quan_Linear, quantize

class BFA(object):

    def __init__(self, criterion, attack_mode='bits'):
        self.attack_mode = attack_mode
        self.criterion = criterion
        self.acc_recorder = []
        self.bit_recorder = []

    def int2bin(self, input, num_bits):
        '''
        convert the signed integer value into unsigned integer (2's complement equivalently).
        '''
        output = input.clone()
        output[input.lt(0)] = 2**num_bits + output[input.lt(0)]
        return output


    def bin2int(self, input, num_bits):
        '''
        convert the unsigned integer (2's complement equivantly) back to the signed integer format
        with the bitwise operations. Note that, in order to perform the bitwise operation, the input
        tensor has to be in the integer format.
        '''
        mask = 2**(num_bits-1) - 1
        output = -(input & ~mask) + (input & mask)
        return output


    def weight_conversion(self, model):
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
                w_bin = self.int2bin(m.weight.data, m.N_bits).char()
                m.weight.data = self.bin2int(w_bin, m.N_bits).float()
        return


    def targeted_bits(self, model, data, target, max_bits=100):
        # # turn on training model for obtaining the gradients
        # model.train()       
        for m in model.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                w_bin = self.int2bin(m.weight.data, m.N_bits).char()



        # # turn on training model for obtaining the gradients
        # model.train()

        # while  max(self.bit_recorder) < max_bits:
        #     output = model(data)
        #     loss = self.criterion(output, target) 
            
            # perform the gradient zero-out before backward to get the new gradient
            for m in model.modules():
                if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                    if m.weight.grad is not None:
                        m.weight.grad.data.zero_() 

        #     loss.backward()






# def BFA(model, data_loader, criterion, targeted_bits=None, targeted_acc=None, use_cuda=True):
#     '''
#     This function performs the Bit-Flip Attack (BFA)
#     Args:
#     '''
#     acc_recorder = []
#     bit_recorder = []
#     # turn on training model for obtaining the gradients
#     model.train()
    
#     if targeted_bits is not None:
#         while max(bit_recorder) < targeted_bits:
#             for i, (input, target) in enumerate(data_loader):           
#                 if use_cuda:
#                     target.cuda(async=True)
#                     input.cuda()
#                 output = model(input)
#                 BFA_loss = criterion(output, target)

#                 # zero-out the gradient before backward
#                 for name, param in model.named_parameters():
#                     param.grad.data.zero_() if param.grad is not None

            

