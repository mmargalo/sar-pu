import matplotlib.pyplot as plt
import torch

def plot_grad_flow(model):
    mod = model.module if torch.cuda.device_count() > 1 else model
    ave_grads = []
    layers = []
    for n, p in mod.named_parameters():
        if(p.requires_grad) and ("bias" not in n) and ("bn" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())

    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig("gradients.png")
   