# SPDX-FileCopyrightText: 2025 Qingcheng.AI
#
# SPDX-License-Identifier: Apache-2.0

def compute_layer_dist_in_pp(num_layers: int, pp_size: int):
    num_layers_of_each_rank = [
        num_layers // pp_size + (1 if i < num_layers % pp_size else 0)
        for i in range(pp_size)
    ]
    # If non-divisible, make the fisrst and the last rank to have fewer layers, because they have pre-layers and post-layers
    if pp_size > 2 and num_layers_of_each_rank[0] > num_layers_of_each_rank[-2]:
        num_layers_of_each_rank[0] -= 1
        num_layers_of_each_rank[-2] += 1
    
    return num_layers_of_each_rank


