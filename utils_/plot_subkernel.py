def ternarize_weight(weight, th_factor, max_w):
    '''
    This function will fetch the threshold tensor/list embedded
    within the quantized convolution/linear layers, then do the
    same math to return the subkernels after quantization.

    This will make the visulization easier.
	'''
    threshold = th_factor*max_w
    scale = weight[weight.ge(threshold)+weight.le(-threshold)].abs().mean()

    weight_tmp = weight.clone().zero_()
    weight_tmp[weight.ge(threshold)] = scale
    weight_tmp[weight.le(-threshold)] = -scale 
    return weight_tmp
    