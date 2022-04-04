import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

# plt.interactive(False)

img = mpimg.imread('/Users/chenweijia/Downloads/600px-Breadth-first-tree.png')
# plt.interactive(False)
# plt.figure("screen", figsize=(20, 5))
# x = np.arange(0, 5, 0.1)
# y = np.sin(x)
# plt.plot(x, y)
a = np.diag(range(100))
imgplot = plt.imshow(a)
status = plt.show()
print status

print '2'
