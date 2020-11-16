import torch


dtype = None
device = None
n_steps = None
partial_a = None
syn_a = None
tau_s = None

def init(dty, dev, n_t, ts):   # a(t_k) = (1/tau)exp(-(t_k-t_m)/tau)H(t_k-t_m)
    global dtype, device, n_steps, partial_a, syn_a, tau_s, stat_flag,\
    output_ori_dict, output_new_dict, memb_p_ori_dict, memb_p_new_dict
    dtype = dty
    device = dev
    n_steps = n_t
    tau_s = ts
    stat_flag  = False
    output_ori_dict = {}
    output_new_dict = {}
    memb_p_ori_dict = {}
    memb_p_new_dict = {}
    partial_a = torch.zeros((1, 1, 1, 1, n_steps, n_steps), dtype=dtype).to(device)
    for t in range(n_steps):
        if t > 0:
            partial_a[..., t] = partial_a[..., t - 1] - partial_a[..., t - 1] / tau_s 
        partial_a[..., t, t] = 1/tau_s
    syn_a = partial_a.clone()
    partial_a /= tau_s 
    for t in range(n_steps):
        partial_a[..., t, t] = -1/tau_s
