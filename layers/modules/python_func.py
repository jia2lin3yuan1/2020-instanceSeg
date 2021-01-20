import numpy as np
from skimage import measure as skmeasure

def create_pairwise_conv_kernel(kernel_size,
                                center_size=16,
                                dia_stride=4,
                                random_dia=False, random_tot_1=1024,
                                return_neiCnt=False, diff_kernel=True):
    '''
    @func: create 1 x fH x fW x num_ch kernel for conv2D to compute pairwise-diff
    @param: kernel_size --[filter_ht, filter_wd]
            center_size -- scalar, neighbours in this window is all considered.
            dia_stride -- scalar, consider dialated pairs on further sides.
            random_dia -- perform random choice to assign neighbour
            random_tot_1 -- # of 1 in random result
            return_neiCnt -- return # of selected neighbour
    '''
    # selected neighbours for computing pairwise difference
    selected_neiI  = np.zeros(kernel_size)
    axis_x, axis_y = np.meshgrid(range(kernel_size[1]), range(kernel_size[0]))
    cy, cx         = kernel_size[0]//2, kernel_size[1]//2
    dist_mat_x     = np.abs(axis_x-cx)
    dist_mat_y     = np.abs(axis_y-cy)
    selected_neiI[(dist_mat_x+dist_mat_y)<center_size] = 1
    if random_dia == False:
        flagI = (np.mod(dist_mat_x, dia_stride) + np.mod(dist_mat_y, dia_stride)) == 0
        #flagI = (np.mod(dist_mat_x+dist_mat_y, dia_stride)==0)
        selected_neiI[flagI] = 1
    else:
        prob_1 = float(random_tot_1) / (kernel_size[0]*kernel_size[1])
        np.random.seed(17929104)
        random_neiI = np.random.choice([0,1], size=kernel_size, p=[1-prob_1, prob_1])
        selected_neiI[random_neiI==1] = 1
    selected_neiI[cy, cx:] = 1
    selected_neiI[cy:, cx] = 1
    selected_neiI[cy, cx]  = 0

    # label each neighbour with continuous indices
    identity_label = axis_y * kernel_size[1] + axis_x
    selected_neiI  = identity_label * selected_neiI

    # remove duplicate pairwise
    selected_neiI[:cy, :]       = 0
    selected_neiI[:cy+1, :cx+1] = 0

    # convert to one hot kernel
    label_neiI = skmeasure.label(selected_neiI)
    label_neiI = np.reshape(label_neiI, [-1])
    kernel     = np.eye(label_neiI.max()+1)[label_neiI]
    kernel     = np.reshape(kernel, [1, kernel_size[0], kernel_size[1], -1])

    if diff_kernel:
        kernel     = 0 - kernel[..., 1:]
        kernel[0, cy, cx, :] = 1

    if return_neiCnt:
        return kernel, label_neiI.max()
    else:
        return kernel


if __name__=='__main__':
    import pdb; pdb.set_trace()
    ta, nei_cnt_0 = create_pairwise_conv_kernel([129, 193], 12,
                                                dia_stride=2,
                                                return_neiCnt=True)

    tb, nei_cnt_1 = create_pairwise_conv_kernel([100, 100], 12,
                                                random_dia=True,
                                                random_tot_1=1024,
                                                return_neiCnt=True)


