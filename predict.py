import global_v as glv
import torch


def Pattern_change_predict(outputs, u, k):
    shape = outputs.shape
    time_steps = glv.n_steps
    neuron_num = shape[0] * shape[1] * shape[2] * shape[3]
    outputs = outputs.view(neuron_num, shape[4])
    u = u.view(neuron_num, shape[4])
    threshold = glv.threshold_dict[k]
    flip_dist = torch.clamp(torch.abs(threshold-u),0,threshold)
    min_index = torch.argmin(flip_dist, dim=-1)
    new_output = outputs > 0
    if k not in glv.dims_dict:
        glv.dims_dict[k] = torch.tensor(list(range(neuron_num)), device = glv.device)
    dim1 = glv.dims_dict[k]
    near_by = flip_dist[dim1, min_index] < 2.0 # Hard threshold guess change
    while near_by.any() == True:
        new_output[dim1, min_index] = new_output[dim1, min_index] ^ near_by
        near_by = near_by & (min_index != time_steps - 1)
        min_index = torch.clamp(min_index+1, 0, time_steps - 1)
        nopp = new_output[dim1, min_index-1] # New output of the previous time step
        mbp = u[dim1, min_index] # Membrane potential of the current time step
        near_by = near_by & ((near_by&(nopp==1)&((1<=mbp)&(mbp<1.8)))|\
                (near_by&(nopp==0)&((0.2<mbp)&(mbp<1))))
    return new_output, outputs

def Accuracy_stat(predict, answer, ori_output):
    shape = answer.shape
    neuron_num = shape[0] * shape[1] * shape[2] * shape[3]
    answer = answer.view(neuron_num, -1)
    ori_output = ori_output.view(neuron_num, -1)
    predict = predict > 0
    answer = answer > 0
    ori_output = ori_output > 0
    predict_correcct = (torch.sum(predict == answer, axis=1) == shape[-1])
    changed_ans = (torch.sum(ori_output == answer, axis=1) < shape[-1])
    predict_changed_correct = torch.sum(predict_correcct.float() *
            changed_ans.float())/torch.sum(changed_ans).float()

#    changed_ans = (torch.sum(ori_output == answer, axis=1) < shape[-1])
#    changed_num = torch.sum(changed_ans, dtype = glv.dtype)
#    predict_change = torch.sum(ori_output == predict, axis=1) < shape[-1]
#    predict_num = torch.sum(predict_change, dtype = glv.dtype)
#    currect_predict_per_change = torch.sum(predict_change & changed_ans,\
#            dtype=glv.dtype)/changed_num
#    currect_predict_per_predict = torch.sum(predict_change & changed_ans,\
#            dtype=glv.dtype)/predict_num
#    correct = torch.sum(torch.sum(answer == predict, axis=1) == shape[-1],\
#            dtype = glv.dtype)
    return predict_changed_correct


def Spike_train_predict():
    output_ori_dict = glv.output_ori_dict
    output_new_dict = glv.output_new_dict
    memb_p_ori_dict = glv.memb_p_ori_dict
    memb_p_new_dict = glv.memb_p_new_dict
    for k in output_ori_dict.keys():
        predict,_ = Pattern_change_predict(output_ori_dict[k],\
                memb_p_ori_dict[k], k)
        glv.accuracy_stat_dict[k] = Accuracy_stat(predict, output_new_dict[k],\
                output_ori_dict[k])
    return

def classify_changes(outputs, u, k):
    predict, outputs = Pattern_change_predict(outputs, u, k)
    time_steps = glv.n_steps
    predict = predict.T > 0
    outputs = outputs.T > 0
    #dim1 = glv.dims_dict[k]
    changes = predict ^ outputs
    for i in range(time_steps-1):
        spike_move = changes[i] & changes[i+1] & (outputs[i+1] ^ outputs[i]) &\
            (predict[i+1] ^ predict[i])
        changes[i] = changes[i] & (~spike_move)
        changes[i+1] = changes[i+1] & (~spike_move)
    return changes.view(u.shape).float()
        #if k+str(i) not in glv.dims_dict:
        #    glv.dims_dict[k+str(i)] = i * torch.ones(len(dim1),device=glv.devic
