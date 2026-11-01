import torch
from torch import nn
from torch.nn import functional as F


class OpinionEncoder(nn.Module):
    def __init__(self,vocab_size,dim,bert_path=None):
        super().__init__();self.bert_path=bert_path
        if bert_path:
            from transformers import AutoModel
            self.bert=AutoModel.from_pretrained(bert_path,local_files_only=True)
            for parameter in self.bert.parameters():parameter.requires_grad=False
            self.project=nn.Sequential(nn.Linear(self.bert.config.hidden_size,dim),nn.ReLU())
        else:
            self.embedding=nn.Embedding(vocab_size,dim,padding_idx=0)
            self.project=nn.Sequential(nn.Linear(dim,dim),nn.ReLU())

    def forward(self,ids):
        mask=ids.ne(0);valid=mask.any(-1)
        if self.bert_path:
            self.bert.eval()
            flat=ids.reshape(-1,ids.shape[-1]);flat_mask=mask.reshape_as(flat);keep=flat_mask.any(-1)
            with torch.no_grad():
                encoded=self.bert(input_ids=flat[keep],attention_mask=flat_mask[keep].long()).last_hidden_state[:,0]
            result=encoded.new_zeros((flat.shape[0],encoded.shape[-1]));result[keep]=encoded
            features=result.reshape(ids.shape[0],ids.shape[1],-1)
        else:
            embedded=self.embedding(ids)
            features=(embedded*mask.unsqueeze(-1)).sum(-2)/mask.sum(-1,keepdim=True).clamp_min(1)
        return self.project(features)*valid.unsqueeze(-1),valid


class MFB(nn.Module):
    def __init__(self,dim,output_dim,factors):
        super().__init__();self.output_dim=output_dim;self.factors=factors
        self.left=nn.Linear(dim,output_dim*factors,bias=False)
        self.right=nn.Linear(dim,output_dim*factors,bias=False)

    def forward(self,left,right):
        product=self.left(left)*self.right(right)
        pooled=product.reshape(-1,self.output_dim,self.factors).sum(-1)
        normalized=torch.sign(pooled)*torch.sqrt(torch.abs(pooled)+1e-8)
        return F.normalize(normalized,dim=-1,eps=1e-8)


class SentimentNetwork(nn.Module):
    def __init__(self,vocab_size,dim=64,heads=4,factors=4,max_opinions=5,variant="full",dropout=.1,bert_path=None):
        super().__init__()
        if dim%heads or dim<4:raise ValueError("Dimension must be divisible by attention heads")
        if variant not in {"full","no_cross","no_mfb","recency_only","popularity_only"}:raise ValueError("Unknown variant")
        self.variant=variant;self.max_opinions=max_opinions
        self.encoder=OpinionEncoder(vocab_size,dim,bert_path)
        self.recency_cls=nn.Parameter(torch.randn(1,1,dim)*.02)
        self.popularity_cls=nn.Parameter(torch.randn(1,1,dim)*.02)
        self.position=nn.Embedding(max_opinions+1,dim)
        self.stage_one=nn.MultiheadAttention(dim,heads,dropout=dropout,batch_first=True)
        self.stage_two=nn.MultiheadAttention(dim,heads,dropout=dropout,batch_first=True)
        self.norm_one=nn.LayerNorm(dim);self.norm_two=nn.LayerNorm(dim)
        self.recency_transformer=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim,heads,dim*2,dropout,batch_first=True),num_layers=1,enable_nested_tensor=False)
        self.popularity_transformer=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim,heads,dim*2,dropout,batch_first=True),num_layers=1,enable_nested_tensor=False)
        self.mfb=MFB(dim,dim,factors)
        self.concat=nn.Sequential(nn.Linear(dim*2,dim),nn.ReLU())
        self.classifier=nn.Sequential(nn.Dropout(dropout),nn.Linear(dim,3))

    def sequence(self,ids,cls):
        values,valid=self.encoder(ids)
        if values.shape[1]>self.max_opinions:raise ValueError("Too many opinions for positional embeddings")
        values=torch.cat([cls.expand(len(values),-1,-1),values],dim=1)
        valid=torch.cat([torch.ones((len(valid),1),dtype=torch.bool,device=ids.device),valid],dim=1)
        values=values+self.position(torch.arange(values.shape[1],device=ids.device)).unsqueeze(0)
        return values,~valid

    def forward(self,recency,popularity,return_attention=False):
        f,fmask=self.sequence(recency,self.recency_cls)
        h,hmask=self.sequence(popularity,self.popularity_cls)
        first=second=None
        if self.variant in {"full","no_mfb"}:
            g,first=self.stage_one(h,f,f,key_padding_mask=fmask,need_weights=return_attention,average_attn_weights=False)
            g=self.norm_one(h+g)
            refined,second=self.stage_two(g,h,h,key_padding_mask=hmask,need_weights=return_attention,average_attn_weights=False)
            refined=self.norm_two(g+refined)
            f=self.recency_transformer(refined,src_key_padding_mask=hmask)
        else:f=self.recency_transformer(f,src_key_padding_mask=fmask)
        h=self.popularity_transformer(h,src_key_padding_mask=hmask)
        fcls,hcls=f[:,0],h[:,0]
        if self.variant=="recency_only":joint=fcls
        elif self.variant=="popularity_only":joint=hcls
        elif self.variant=="no_mfb":joint=self.concat(torch.cat([fcls,hcls],dim=-1))
        else:joint=self.mfb(fcls,hcls)
        logits=self.classifier(joint)
        if return_attention:return logits,{"trending_to_timely":first,"attended_to_trending":second}
        return logits
