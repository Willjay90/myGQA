import numpy as np
import torch
import math, copy, time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DefaultImageFeature(nn.Module):
    def __init__(self, in_dim):
        super(DefaultImageFeature, self).__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim

    def forward(self, image):
        return image

class EncoderDecoder(nn.Module):
    "standord encoder-decoder architecture."
    def __init__(self, encoder, decoder, src_embed, tgt_embed, img_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.img_embed = img_embed
        self.generator = generator

    def forward(self, src, tgt, image_feat, src_mask, tgt_mask):
        res = self.generator(self.decode(self.encode(src, image_feat, src_mask), image_feat, src_mask, tgt, tgt_mask))
        return res
        # return self.decode(self.encode(src, image_feat, src_mask), image_feat, src_mask, tgt, tgt_mask)
    def encode(self, src, image_feat, src_mask):
        return self.encoder(self.src_embed(src), image_feat, src_mask)
    def decode(self, memory, image_feat, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, image_feat, src_mask, tgt_mask)

# Generate Answer
class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        x = x.float()
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2

class SubLayerConnection(nn.Module):
    """
    A residual connection followed by layernorm.
    """
    def __init__(self, size, dropout):
        super(SubLayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, image_feat, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, image_feat, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    "Encoder is made up of self attention and feed forward."
    def __init__(self, size, self_attn, img_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.img_attn = img_attn
        self.sublayer = clones(SubLayerConnection(size, dropout), 3) # 2 layers encoder
        self.size = size

    def forward(self, x, image_feat, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, image_feat, mask))    # multi-head attn
        x = self.sublayer[1](x, lambda x: self.img_attn(x, image_feat))     # image_attn
        tmp = self.sublayer[1](x, self.feed_forward)
        return self.sublayer[2](x, self.feed_forward)                       # feed-forward
    
class ImageAttention(nn.Module):
    def __init__(self, img_dim, size):
        super(ImageAttention, self).__init__()
        self.fc_layer = nn.Linear(img_dim, size)

    def forward(self, memory, img_feature):
        batch_size = img_feature.size(0)
        img = img_feature.view(batch_size, -1)
        
        x = self.fc_layer(img)
        x = torch.unsqueeze(x, 1)
        x = x.repeat(1, memory.size(1), 1)

        scores = torch.matmul(memory, x.transpose(-2, -1))
        img_attn = F.softmax(scores, dim=-1)

        return torch.matmul(img_attn, memory)
        

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x, memory, image_feat, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, image_feat, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward."
    def __init__(self, size, self_attn, src_attn, img_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.img_attn = img_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SubLayerConnection(size, dropout), 4) # 3 layers decoder
    
    def forward(self, x, memory, image_feat, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, image_feat, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        x = self.sublayer[2](x, lambda x: self.img_attn(x, image_feat))     # image_attn
        return self.sublayer[3](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, image_feat, mask=None, dropout=None):
    'Scale Dot-Product as described in Transformer'
    '**image_feat** as input for further engineering.'
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # batch * h * max_len * max_len 

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, image_feat, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k  (8 x 64)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, image_feat, mask=mask, dropout=self.dropout) # nbatch x h x len x d_k

        # 3) "Concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)
    
class PositionwiseFeedForward(nn.Module):
    """
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


## Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()) * -(math.log(10000.0) / d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x) # get rid of nan
        return self.dropout(x)

