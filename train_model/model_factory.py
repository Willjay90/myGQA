import torch.nn as nn
import torch
import copy
from train_model.transformer import *

def make_model(src_vocab, tgt_vocab, img, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        nn.Sequential(DefaultImageFeature(img), c(position)),
        Generator(d_model, tgt_vocab)
    )

    # Initialize parameters with Glorot / fan_avg. 
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    if torch.cuda.is_available():
        model = model.cuda()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return model