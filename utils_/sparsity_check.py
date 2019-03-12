from models.tern_threshold_trainable_5 import _quanFunc

def Sparsity_check(module):
    """
    Return the sparsity of the input torch.tensor
    sparsity = #zero_elements/#tot_elements
    """
    kernel_tmp =  _quanFunc(module.mu.data, module.th_clip[0].data, module.scale_factor[0].data)(module.weight.data).cpu()
    num_tot = kernel_tmp.data.numpy().size
    num_zeros = kernel_tmp.data.eq(0).sum().item()
    sparsity = num_zeros / num_tot * 100
    return sparsity