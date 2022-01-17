from cProfile import label
from turtle import color
import numpy as np
import torch
import matplotlib.pyplot as plt

COLORN=["red", "orange", "blue"]

def visualize(pred, label):
    """
    (T, #spk)
    """

    fig = plt.figure(1)
    y_locator = plt.MultipleLocator(1)

    ax1=plt.subplot(2,1,1)
    plt.gca().get_yaxis().set_major_locator(y_locator)
    plt.ylim(0.9, 3.1)
    for i in range(pred.shape[1]):
        each_spk = np.nonzero(pred[:, i])[0]
        if len(each_spk) == 0:
            continue
        segments = []
        st, ed = each_spk[0], each_spk[0]     
        for idx in each_spk[1:]:
            if idx - ed > 1:
                segments += [np.arange(st, ed+1), np.ones(ed+1-st)*(i+1)]
                st = idx
            ed = idx
        segments += [np.arange(st, ed+1), np.ones(ed+1-st)*(i+1)]
        print(segments)
        plt.setp(plt.plot(*segments), color=COLORN[i])

    ax2=plt.subplot(2,1,2)
    plt.gca().get_yaxis().set_major_locator(y_locator)
    plt.ylim(0.9, 3.1)
    for i in range(label.shape[1]):
        each_spk = np.nonzero(label[:, i])[0]
        if len(each_spk) == 0:
            continue
        segments = []
        st, ed = each_spk[0], each_spk[0]
        for idx in each_spk[1:]:
            if idx - ed > 1:
                segments += [np.arange(st, ed+1), np.ones(ed+1-st)*(i+1)]
                st = idx
            ed = idx
        segments += [np.arange(st, ed+1), np.ones(ed+1-st)*(i+1)]
        plt.setp(plt.plot(*segments), color="black")
    # plt.show()
    plt.savefig("./test.png")
    plt. close(1)
    print(pred)
    print(pred.shape)
    print(label)
    print(label.shape)
    input()


if __name__=="__main__":
    x = torch.randn((500,3))
    pred = (x > 0.4).float().numpy()
    label = (x > 0.6).float().numpy()
    visualize(pred, label)

