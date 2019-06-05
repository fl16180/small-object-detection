import os
import pandas as pd

LOG = '../saved/log/VOC_SSD/0601_052820/info.log'

def load_log(logfile):
    epochs = []
    train_loss = []
    val_loss = []

    with open(LOG, 'r') as f:
        for l in f:
            if '  epoch  ' in l:
                tmp = l.split(':')[-1]
                tmp = "".join(tmp.split())
                epochs.append(int(tmp))
            elif '  loss  ' in l:
                tmp = l.split(':')[-1]
                tmp = "".join(tmp.split())
                train_loss.append(float(tmp))
            elif '  val_loss  ' in l:
                tmp = l.split(':')[-1]
                tmp = "".join(tmp.split())
                val_loss.append(float(tmp))

    cols = {'epoch': epochs, 'train': train_loss, 'val': val_loss}
    loss = pd.DataFrame(cols)

    return loss


if __name__ == '__main__':

    loss = load_log(LOG)
    print(loss.head())
    loss.to_csv('./log.csv', index=False)
