import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from dataclasses import dataclass
import torch.nn.functional as F
import math

@dataclass
class ModelArgs:
    channels : int = 3
    img_size : int = 224
    patch_size : int = 4
    n_layers : int = 1
    out_dim : int = 10
    embd_dim : int  = 756
    n_heads : int = 4


class PatchEmbeddings(nn.Module):
    def __init__(self, in_channels: int = 3, img_size: int = 28, patch_size: int = 16, emb_size: int = 768):
            super().__init__()
            self.patch_size = patch_size
    
            self.projection = nn.Sequential(
                # Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
                # nn.Linear(patch_size * patch_size * in_channels, emb_size)
                
                nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
                Rearrange('b e (h) (w) -> b (h w) e'),
            )
            self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
            self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b,_,_,_ = x.shape

        # B Embed H W -> B Num_patches  EMbed
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.positions
        # B Num_patches + 1  Embed
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, args : ModelArgs) -> None:
        super().__init__()

        # Indicates the number of heads for the keys and values
        self.n_kv_heads = args.n_heads
        # Indicates the number of heads for queries
        self.n_heads_q =  args.n_heads
        # Indicates how many times the heads of keys and Values should be repeated to match the number of heads for queries
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indiactes the dimension of the embedding
        self.head_dim = args.embd_dim // args.n_heads

        self.wq = nn.Linear(args.embd_dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.embd_dim ,self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.embd_dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.embd_dim, bias=False)
        
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, dim = x.shape # (B,1,dim)
 
        # (B,n,dim) -> (B, n , H_Q * Head_dim)
        xq = self.wq(x)
        # (B,n ,dim) -> (B, n, H_KV * Head_dim)
        xk = self.wk(x)
        xv = self.wv(x)
        
        # (B, n, H_Q * Head_dim) -> (B, n, H_Q, Head_dim)
        queries = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, n, H_KV * Head_dim) -> (B, n, H_KV, Head_dim)
        keys = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        values = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

       


        # (B, n, H_Q, Head_dim) --> (B, H_Q, n, Head_dim)
        queries = queries.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)
        # print(f"Queries Shape : {queries.shape}")
        # print(f"Queries Shape : {keys.shape}")
        # print(f"Queries Shape : {values.shape}")

        # (B, H_Q, n, Head_dim) * (B, H_KV, Head_dim, n) -> (B, H_Q, n, n)

        scores = torch.matmul(queries, keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores = torch.softmax(scores, dim=-1)

        # (B, H_Q, n,n) * (B, H_Q , n, Head_dim) -> (B, H_Q, n, Head_dim)
        output = torch.matmul(scores, values)  
        # (B, H_Q, n, Head_dim) -> (B, n, H_Q, Head_dim) -> (B, n, Dim)
        output = output.transpose(1,2).contiguous().view(batch_size,seq_len, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.embd_dim
        hidden_dim = int(2 * hidden_dim / 3)

        self.w1 = nn.Linear(args.embd_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.embd_dim, bias=False)
        self.w3 = nn.Linear(args.embd_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish  =  F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x
     

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.embd_dim
        self.head_dim = args.embd_dim // args.n_heads

        self.attention = MultiHeadAttention(args)
        self.feed_forward = FeedForward(args)


        # Normalization Before Attention
        self.attention_norm = nn.LayerNorm(args.embd_dim)

        # Normalization Before Feed Forward
        self.ff_norm = nn.LayerNorm(args.embd_dim)



    def forward(self, x : torch.Tensor ):

        # (B, Seq_len, DIM) + (B, Seq_len, DIM) -> (B, Seq_len, DIM)
        h = x + self.attention.forward(self.attention_norm(x))
        out = h + self.feed_forward.forward(self.ff_norm(h))
        return out

class ViT(nn.Module):
    def __init__(self, args: ModelArgs):
          super().__init__()

          self.channels = args.channels
          self.height = args.img_size
          self.width = args.img_size
          self.num_patches = ( self.height // args.patch_size) * ( self.width // args.patch_size)
          self.n_layers = args.n_layers 
          self.out_dim = args.out_dim
          self.embd_dim = args.embd_dim
          self.patch_embedding = PatchEmbeddings(in_channels=self.channels,
                                              patch_size=args.patch_size,
                                              emb_size= args.embd_dim)
          self.layers = nn.ModuleList()
         

          for _ in range(self.n_layers):
               self.layers.append(EncoderBlock(args))

           # Classification head
          self.head = nn.Sequential(
               nn.LayerNorm(self.embd_dim), 
               nn.Linear(self.embd_dim, self.out_dim)
               )
    
    def forward( self, img:torch.Tensor):
         
         # B Num_patches + 1  Embed
        h = self.patch_embedding(img)

        b, n, _  = h.shape

        for layer in self.layers :
              h = layer(h)

        output = F.softmax(self.head(h), dim=-1)

        return output[:,0, : ]


         
         
          
        