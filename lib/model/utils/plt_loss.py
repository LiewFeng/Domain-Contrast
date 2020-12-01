import numpy as np
import matplotlib.pyplot as plt
import os

def plt_loss(epoch, dir_, name, value):
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    axis = np.linspace(1,epoch,epoch)
    label = '{}'.format(name)
    fig = plt.figure()
    plt.title(label)
    plt.plot(axis, value)
#     plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('{}/{}.pdf'.format(dir_, name))
    plt.close(fig)