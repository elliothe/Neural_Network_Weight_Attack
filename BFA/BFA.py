
def weight_conversion(model):





def BFA(object):

    def __init__(self, criterion, attack_mode='bits'):
        self.attack_mode = attack_mode
        self.criterion = criterion
        self.acc_recorder = []
        self.bit_recorder = []

    def targeted_bits(self, model, data, target, max_bits=100):
        # turn on training model for obtaining the gradients
        model.train()

        while  max(self.bit_recorder) < max_bits:
            output = model(data)
            loss = self.criterion(output, target) 
            
            # perform the gradient zero-out before backward to get the new gradient

            loss.backward()



def BFA(model, data_loader, criterion, targeted_bits=None, targeted_acc=None, use_cuda=True):
    '''
    This function performs the Bit-Flip Attack (BFA)
    Args:
    '''
    acc_recorder = []
    bit_recorder = []
    # turn on training model for obtaining the gradients
    model.train()
    
    if targeted_bits is not None:
        while max(bit_recorder) < targeted_bits:
            for i, (input, target) in enumerate(data_loader):           
                if use_cuda:
                    target.cuda(async=True)
                    input.cuda()
                output = model(input)
                BFA_loss = criterion(output, target)

                # zero-out the gradient before backward
                for name, param in model.named_parameters():
                    param.grad.data.zero_() if param.grad is not None

            

