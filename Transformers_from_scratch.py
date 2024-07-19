import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"
    
        self.values = nn.Linear(self.head_dim, self.head_dim, bias= False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias= False)
        self.fc_out = nn.Linear(heads * self.head_dim , embed_size)

    def forward(self,values, keys, query, mask):
        N = query.shape[0]  # How many examples we are goingto send at same time
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # split embeddings in self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshsape(N, key_len,self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        ## Sums the product of the elements of the input operands along dimensions specified using a 
        ## notation based on the Einstein summation convention.
        energy = torch.einsum("nqhd,nkhd -> nhqk", [queries, keys])
        
        # insted of torch.einsum you can use torch.bmm - batch metrix multiply
        # queries_shape : (N, query_len, heads, heads_dim)
        # keys_shape : (N, key_len, heads, head_dims)
        # energy_shape : (N, heads, query_len, key_len)
         
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim = 3)
        #this make sure we are normalizing accross kay length

        out = torch.einsum("nhql,nlhd -> nqhd",[attention, values]).reshape(
            N, query_len, self.heads*self.head_dim
        )
        #attention shape : (N, heds, query_len, key_len)
        #value shape : (N, value_len, heads_dim)
        # after einsum  : (N, query_len, heads, head_dim) then flattern last 2 dim

        out = self.fc_out(out) # just mast the embed size to embed size

        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size,heads, dropout, forward_expansion):
        super(TransformerBlock,self).__init__()

        self.attention =SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)

        # layer norm take avarage accross layer and normalize it (per example)
        # batch norm takes avarage accross batch and normalize it (per batch)
        # layer norm has more computetions

        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size), 
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self,value, key, query, mask):
        attention = self.attention(value ,key, query, mask)
        x = self.dropout(self.norm1(attention + query ))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
 
        return out
    
class Encoder(nn.Module):
    def __init__(self,
                 source_vocab_size,
                 embed_size,
                 number_layers,
                 heads,
                 device,
                 forward_extension,
                 dropout,
                 max_length):
        
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(source_vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_length, embed_size)
        self.layers =nn.ModuleList(
            [
                TransformerBlock(embed_size,
                                 heads,
                                 dropout=dropout,
                                 forward_expansion=forward_extension)
            ]
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x,mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N,seq_length).to(self.device)

        self.word_embedding(x) + self.