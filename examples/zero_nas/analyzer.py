# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import math
import random
from random import sample
random.seed(123)


with open('expressivity.pkl', 'rb') as handle:
    e_list = pickle.load(handle)

sorted_by_zen = [ i for i in e_list if not(math.isnan(i[0])) and i[0] != 1000 ]
sorted_by_num_params = sorted([i[1] for i in sorted_by_zen])



def postprocess(field):
    params = []
    for i in field.split(', ')[1:]:
        params.append(i.split(': ')[-1])
    return params


# model params
num_params_dict = {}
for num_params in sorted_by_num_params:
    for fields in sorted_by_zen:
        if fields[1]==num_params:
            print(postprocess(fields[2]))
            if num_params in num_params_dict.keys():
                num_params_dict[num_params].append(fields)
            else:
                num_params_dict[num_params] = [fields]


pp_zen, pp_size, zen_size = [], [], []

# split into data
max_len = max( [ len(v) for k,v in num_params_dict.items() ] )
for k,v in num_params_dict.items():
    if len(v)==max_len:
        zen_size = v
    pp_zen.append(sample(v,1))
    pp_size.append(sample(v,1))


with open('zen_size.pkl', 'wb') as handle:
    pickle.dump(zen_size, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('pp_zen.pkl', 'wb') as handle:
    pickle.dump(pp_zen, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('pp_size.pkl', 'wb') as handle:
    pickle.dump(pp_size, handle, protocol=pickle.HIGHEST_PROTOCOL)
