import torch
import torch.nn as nn
import sys
import os
import time
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


def one_stage_train(data_iter, model, loss_compute):
    start = time.time()
    total_loss = 0

    for i, batch in enumerate(data_iter):
        out = model.forward(batch.stc, )
    