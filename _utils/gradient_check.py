import matplotlib.pyplot as plt
import torch
from os.path import join

def plot_grad_flow(model, name=None, dest="./"):
    mod = model.module if torch.cuda.device_count() > 1 else model
    ave_grads = []
    layers = []
    for n, p in mod.named_parameters():
        if(p.requires_grad) and ("bias" not in n) and ("bn" not in n):
            layers.append(n.replace(".weight",""))
            ave_grads.append(p.grad.abs().mean().cpu())

    plt.figure(name)
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical", fontsize=8)
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(join(dest, name+".png"))
   