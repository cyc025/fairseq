

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import csv

plt.style.use('ggplot')


x = []
y = []
z = []

with open('results/zen_model_perp.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in list(plots)[1:]:
        x.append(float(row[0]))
        y.append(float(row[1]))
        z.append(float(row[2]))

correlation, p_value = scipy.stats.pearsonr(x, y)
plt.plot(x,y,label=f'Correlation: {round(correlation,4)}\nP value: {round(p_value,100)}', marker="x")


# plt.xlabel('Zen-score')
# plt.ylabel('Perplexity')
# plt.title('Zen-score vs. Perplexity')
plt.xlabel('Model Size')
plt.ylabel('Zen-score')
plt.title('Model size vs. Zen-score')

# 
#
# fig, ax = plt.subplots()
# ax.scatter(x, y)

for i, txt in enumerate(z):
    plt.annotate(txt, (x[i], y[i]))


plt.legend()
plt.show()
