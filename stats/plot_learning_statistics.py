import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcdefaults()


def plot_individual_learning_rates(accuracies_sorted):
    _, k = np.shape(accuracies_sorted)
    chains = np.arange(k)
    means = np.mean(accuracies_sorted, axis=0)
    plt.figure(figsize=(20, 15))
    plt.plot(np.swapaxes(accuracies_sorted, 0, 1), 'bo', alpha=1.0/k)
    mean_line, = plt.plot(means, 'k:', label='means')
    plt.legend(handles=[mean_line], loc=1, fontsize=20)
    plt.xticks(chains, chains)
    plt.xlabel('chains', fontsize=20)
    plt.ylabel('accuracy mean', fontsize=20)
    plt.title('learning rates of individual chains', fontsize=20)
    plt.grid(True)
    plt.show()


def polarize_accuracies(accs):
    for i, accuracy in enumerate(accs):
        if accuracy < 0.5:
            accs[i] = 1.0 - accuracy
    return accs


def process_accuracies(accs):
    n, k = np.shape(accs)
    for i in range(n):
        accs[i, :] = np.sort(polarize_accuracies(accs[i, :]))[::-1]
    return accs


def plot_frequency_distribution(d1, d2, d3, d4):
    labels = ['(128x1)-PUF', '(64x1)-PUF', '(32x1)-PUF', '(16x1)-PUF']
    data = np.concatenate((d1, d2, d3, d4), axis=1)
    fig, ax = plt.subplots(figsize=(20, 15))
    plt.hist(data, 100, histtype='bar', label=labels, stacked=True)
    plt.xticks(np.arange(0.5, 0.8, 0.05))
    plt.xlabel('distance', fontsize=20)
    plt.ylabel('frequency', fontsize=20)
    plt.title('frequency distribution of distances', fontsize=20)
    legend = plt.legend(loc='center', shadow=True)
    for label in legend.get_texts():
        label.set_fontsize('large')
    for line in legend.get_lines():
        line.set_linewidth(2)
    plt.axis([0.5, 0.8, 0, 2500])
    ax.text(0.71, 2100, 'This diagram results from distances\n'
                        '- where 0.4 equals 0.6 -\n'
                        'of 10,000 times each 2 randomly sampled PUFs\n'
                        'compared by each 100,000 randomly sampled\n'
                        'challenges for lengths 16, 32, 64 and 128.',
            bbox={'facecolor': 'white', 'pad': 10})
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.8)
    ax.grid(which='minor', linestyle='-', linewidth='0.5', color='grey', alpha=0.5)
    plt.show()


def plot_stability_vs_noisiness(d1, d2, d3, d4):
    left = 0
    right = 2
    top = 1.02
    bottom = 0.48
    fig, ax = plt.subplots(figsize=(20, 15))
    plt.plot([left, right], [0.5, 0.5], 'k--', linewidth=3)
    plt.plot([left, right], [1.0, 1.0], 'k--', linewidth=3)
    plt.xticks(np.arange(left, right+0.1, 0.05), rotation='vertical')
    plt.yticks(np.arange(bottom+0.02, top, 0.02))
    x = np.arange(0.001, 2.001, 0.001)
    plt.plot(x, d1, linewidth=0.5, label='(128x1)-PUF')
    """
    plt.plot(x, d2, linewidth=0.5, label='(64x1)-PUF')
    plt.plot(x, d3, linewidth=0.5, label='(32x1)-PUF')
    plt.plot(x, d4, linewidth=0.5, label='(16x1)-PUF')
    legend = plt.legend(loc='center left', shadow=True)
    for label in legend.get_texts():
        label.set_fontsize('large')
    for line in legend.get_lines():
        line.set_linewidth(2)
    """
    plt.xlabel('noisiness', fontsize=20)
    plt.ylabel('stability', fontsize=20)
    plt.title('stability vs. noisiness', fontsize=20)
    plt.axis([left, right, bottom, top])
    ax.text(1.4, 0.9, 'This diagram results from means of reliabilities\n'
                      'of each 1,000 randomly sampled (128x1)-PUFs\n'
                      'evaluated by each 100,000 randomly sampled\n'
                      'challenges for 2,000 different noisinesses\n'
                      'from 0.001 to 2.',
        bbox={'facecolor':'white', 'pad':10})
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.8)
    ax.grid(which='minor', linestyle='-', linewidth='0.5', color='grey', alpha=0.5)
    plt.show()


# Example
accuracies = np.array([
    [0.95, 0.90, 0.89, 0.60],
    [0.99, 0.98, 0.90, 0.65],
    [0.95, 0.92, 0.89, 0.75],
    [0.98, 0.96, 0.85, 0.69],
    [0.97, 0.93, 0.88, 0.63],
    [0.96, 0.93, 0.93, 0.79],
    [0.99, 0.99, 0.89, 0.78],
    [0.99, 0.99, 0.98, 0.80],
    [0.98, 0.97, 0.89, 0.66],
    [0.99, 0.96, 0.95, 0.51],
])

raw_distances = np.array(
    [.51, .55, .53, .6, .68, .55, .78, .53, .5, .53]
)


# Executions
#plot_individual_learning_rates(accuracies)
"""
# plot stability vs noisiness
path1 = '~/workspace/some_data/intra/results/intra_128.csv'
path2 = '~/workspace/some_data/intra/results/intra_64.csv'
path3 = '~/workspace/some_data/intra/results/intra_32.csv'
path4 = '~/workspace/some_data/intra/results/intra_16.csv'
df1 = pd.read_csv(path1, header=None)
df2 = pd.read_csv(path2, header=None)
df3 = pd.read_csv(path3, header=None)
df4 = pd.read_csv(path4, header=None)
plot_stability_vs_noisiness(df1.values, df2.values, df3.values, df4.values)
"""

#"""
# plot frequency distribution
path1 = '~/workspace/some_data/dist/distances_128.csv'
path2 = '~/workspace/some_data/dist/distances_64.csv'
path3 = '~/workspace/some_data/dist/distances_32.csv'
path4 = '~/workspace/some_data/dist/distances_16.csv'
df1 = pd.read_csv(path1, header=None)
df2 = pd.read_csv(path2, header=None)
df3 = pd.read_csv(path3, header=None)
df4 = pd.read_csv(path4, header=None)
plot_frequency_distribution(df1.values, df2.values, df3.values, df4.values)
#"""

"""
# plot chain's distribution
path1 = '~/workspace/some_data/logs/16p-result.log'
path2 = '~/workspace/some_data/logs/32p-result.log'
path3 = '~/workspace/some_data/logs/64p-result.log'
path4 = '~/workspace/some_data/logs/128p-result.log'
df1 = pd.read_csv(path1, sep='\t', header=None)
df2 = pd.read_csv(path2, sep='\t', header=None)
df3 = pd.read_csv(path3, sep='\t', header=None)
df4 = pd.read_csv(path4, sep='\t', header=None)

data = np.array(df1.values[:, 10])
lines = len(data)
matrix = np.zeros((lines, 4))
for i, array in enumerate(data):
    a = array.split(',')
    for j, acc in enumerate(a):
        val = float(acc)
        matrix[i, j] = val

#for i, accs in enumerate(matrix):
#    matrix[i, :] = polarize_accuracies(accs)
#for i, accs in enumerate(matrix):
#    matrix[i, :] = process_accuracies(accs)
processed = process_accuracies(matrix)
plot_individual_learning_rates(processed)
"""
