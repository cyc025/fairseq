

import os
import re
import matplotlib.pyplot as plt
import numpy as np
import sys

plt.style.use('ggplot')





def get_mean_data(directory):

    indices = []
    values_dict = {}
    mean_dict = {}
    std_dict = {}

    for filename in os.listdir(directory):
        index = re.search('profile_(.*).log', filename).group(1)
        print(index,filename)
        if int(index)>100:
            break
        try:
            values = [float(value.strip().split('ms')[0]) for value in open(f'{directory}/{filename}','r')]
        except:
            values = [float(value.strip().split('s')[0]) for value in open(f'{directory}/{filename}','r')]
        indices.append(int(index))
        values_dict[index] = np.array(values)
        mean_dict[index] = np.mean(values_dict[index])
        std_dict[index] = np.std(values_dict[index])

    x_pos, CTEs, error = [], [], []
    for index in sorted(indices):
        x_pos.append(index)
        CTEs.append(mean_dict[str(index)])
        error.append(std_dict[str(index)])

    return CTEs


width = 0.8

step_1 = get_mean_data(sys.argv[1])
step_2 = get_mean_data(sys.argv[2])
step_4 = get_mean_data(sys.argv[3])


indices = np.arange(len(step_2))

plt.bar(indices, step_1, width=width,
         label='1-step')
plt.bar([i+0.25*width for i in indices], step_2,
        width=0.5*width, label='2-step')
plt.bar([i+0.25*width for i in indices], step_4,
        width=0.5*width, label='4-step')

plt.xticks(indices+width/2.,
           ['{}'.format(i) for i in range(len(step_2))] )

plt.legend()

plt.show()
