import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

x1 = [
    0.4465706130316,0.4427301580527,0.4396503370340,0.4534179859091,0.4523408728843,
]
x2= [
0.7512873704864,1.4668953958379,2.3420869251651,3.0900747959414,4.2299929718525
]

# x1 = [0.1499195667908,0.1482124272037,0.1429878847885,0.1343424181963,0.1189820538468,]
# x2 = [0.0869895283544,0.0827033414760,0.0997799188867,0.1188327421923,0.1323059125443]
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.boxplot([x1, x2,], labels=['Gaussian', 'Laplace'])
ax.set_xlabel('Noise types',fontsize=18)
ax.set_ylabel('Hausdorff Dist.',fontsize=18)
ax.set_ylim(0.4, 4.3)

plt.show()