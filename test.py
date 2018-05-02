import numpy as np

v = np.array([1,2,3,4])
u = np.array([-1,-2,-3,-4])
diff = np.subtract(u,v)
res = np.linalg.norm(diff)
print(res)


mat = np.array([
    [1,2,3,4],
    [5,6,7,8]
])

#print(len(mat))



import matplotlib.pyplot as plt
import numpy as np

data = [np.random.normal(0, std, 1000) for std in range(1, 6)]

box = plt.boxplot(data, notch=True, patch_artist=True)

colors = ['cyan', 'lightblue', 'lightgreen', 'tan', 'pink']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

plt.show()