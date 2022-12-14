import torch
import numpy as np

def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]