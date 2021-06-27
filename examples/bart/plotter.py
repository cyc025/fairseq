

import os
import re
import numpy as np
import numpy as np
import matplotlib.pyplot as plt


plt.style.use('ggplot')


indices = []
values_dict = {}
mean_dict = {}
std_dict = {}
directory = "results"
for filename in os.listdir(directory):
    index = re.search('profile_(.*).log', filename).group(1)
    try:
        values = [float(value.strip().split('ms')[0]) for value in open(f'results/{filename}','r')]
    except:
        values = [float(value.strip().split('s')[0]) for value in open(f'results/{filename}','r')]
    indices.append(int(index))
    values_dict[index] = np.array(values)
    mean_dict[index] = np.mean(values_dict[index])
    std_dict[index] = np.std(values_dict[index])


x_pos, CTEs, error = [], [], []
for index in sorted(indices):
    x_pos.append(index)
    CTEs.append(mean_dict[str(index)])
    error.append(std_dict[str(index)])


# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Latency')
ax.set_xlabel('Sequence length (#. of tokens)')
ax.set_xticks(sorted(indices))
ax.set_xticklabels(x_pos)
ax.set_title('A Plot of Latency vs. Sequence Length')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('latency_length.png')
plt.show()
