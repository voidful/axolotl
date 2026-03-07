import sys
# Simulating the exact path that loss takes in evaluation_loop
def simulate_eval_loop(loss_tensor):
    import numpy as np
    import torch
    all_losses = []
    
    # 1. prediction_step returns (loss_tensor, None, None)
    # 2. evaluation_loop nested_gather
    from torch.distributed import is_initialized
    if loss_tensor is not None:
        loss_tensor = loss_tensor.repeat(4) # batch size
        all_losses.append(loss_tensor)
        
    if all_losses:
        concat = torch.cat(all_losses)
        print("concatenated shape:", concat.shape)
        # 3. axolotl's patched mean
        print("nanmean:", np.nanmean(concat.cpu().numpy()).item())

import torch
simulate_eval_loop(torch.tensor(0.5))
