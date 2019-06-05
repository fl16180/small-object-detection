import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../log.csv')

x = df.epoch
train = df.train
val = df.val

fig, ax = plt.subplots(ncols=3)

ax[0].plot(x, train, label='train')
ax[0].plot(x, val, label='val')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].set_title('SSD300')
ax[0].legend()
plt.show()
