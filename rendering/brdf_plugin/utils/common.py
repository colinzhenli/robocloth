import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def get_mgrid(sidelen, dim=1):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def minmax_scale(x):
    return (2 * x - x.min() - x.max()) / (x.max() - x.min())


def create_directory(logdir):
    try:
        os.makedirs(logdir)
    except FileExistsError:
        pass


def interpolate(x, y, lamb=0.2):
    return (x - y) * lamb + y


def get_device():
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def plot_tsne(arrs, fname=None, colors=None, alpha=0.3, labels=None):  # arrs = [arr1, arr2, ...]
    tsne = TSNE(n_components=2, init='random')
    X_emb = tsne.fit_transform(np.concatenate(arrs))

    idx0, idxf = 0, 0
    for iarr, arr in enumerate(arrs):
        idx0 = idxf
        idxf = idxf + len(arr) - 1
        color = None if colors is None else colors[iarr]
        label = None if labels is None else labels[iarr]
        plt.scatter(X_emb[idx0:idxf, 0], X_emb[idx0:idxf, 1], alpha=alpha, color=color, label=label)
    plt.legend()
    plt.xlabel("X_t-SNE")
    plt.ylabel("Y_t-SNE")
    if fname is not None:
        plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()


def load_model_torch(pt):
    model = torch.jit.load(pt)
    return model


def normalize_phid(orig_phid):
    phid = orig_phid.copy()
    phid = np.where(phid < 0, phid + 2 * np.pi, phid)
    phid = np.where(phid >= 2 * np.pi, phid - 2 * np.pi, phid)
    return phid


def mask_from_array(arr):
    if (len(arr.shape) > 1):
        mask = np.linalg.norm(arr, axis=1)
        mask[mask != 0] = 1
    else:
        mask = np.where(arr != 0, 1, 0)
    return mask
