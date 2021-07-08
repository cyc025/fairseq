

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import csv

plt.style.use('ggplot')


x = []
y = []

with open('results/model_zen.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in list(plots)[1:]:
        x.append(float(row[0]))
        y.append(float(row[1]))

correlation, p_value = scipy.stats.pearsonr(x, y)
plt.plot(x,y,label=f'Correlation: {round(correlation,4)},\nP value: {round(p_value,100)}', marker="x")


# plt.xlabel('Zen-score')
# plt.ylabel('Perplexity')
# plt.title('Zen-score vs. Perplexity')
plt.xlabel('Model Size')
plt.ylabel('Zen-score')
plt.title('Model size vs. Zen-score')


plt.legend()
plt.show()
