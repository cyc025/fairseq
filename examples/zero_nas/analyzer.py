# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle

with open('expressivity.pkl', 'rb') as handle:
    e_list = pickle.load(handle)

filtered_e_list = len([ i for i in expressivity_list if not(math.isnan(i[0])) and i[0] != 1000 ])

print(filtered_e_list)
