import re
from collections import Counter
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

LABELS = ["negative", "neutral", "positive"]
TEXT = {
    -1: ["earnings decline", "cash pressure", "demand weakness", "margin contraction"],
    0: ["guidance unchanged", "routine filing", "steady operations", "mixed outlook"],
    1: ["earnings growth", "strong cash flow", "demand recovery", "margin expansion"]}


def simulate(companies=600,seed=31):
    if companies<60:raise ValueError("At least 60 companies required")
    rng=np.random.default_rng(seed);samples=[]
    for company in range(companies):
        for window in range(2):
            recency=int(rng.choice([-1,0,1]));popularity=int(rng.choice([-1,0,1]))
            score=.7*recency+.3*popularity+float(rng.normal(0,.18))
            label=0 if score<-.35 else 2 if score>.35 else 1
            as_of=pd.Timestamp("2025-06-01",tz="UTC")+pd.Timedelta(days=window*30)
            record={"sample_id":f"S{company:04d}_{window}","company_id":f"C{company:04d}",
                    "as_of":as_of.isoformat(),"label":label,"recency":[],"popularity":[]}
            for stream,polarity in [("recency",recency),("popularity",popularity)]:
                for i in range(int(rng.integers(1,6))):
                    observed=polarity if rng.random()>.18 else int(rng.choice([-1,0,1]))
                    phrase=str(rng.choice(TEXT[observed]))
                    text=(f"Company reports {phrase} in the quarterly business update."
                          if stream=="recency" else f"Investors discuss {phrase}; market reaction remains in focus.")
                    published=as_of-pd.Timedelta(hours=int(rng.integers(3,240)))
                    record[stream].append({"opinion_id":f"{record['sample_id']}_{stream}_{i}",
                        "text":text,"published_at":published.isoformat(),
                        "available_at":(published+pd.Timedelta(hours=1)).isoformat(),
                        "engagement":int(rng.integers(1,500)),
                        "engagement_observed_at":(as_of-pd.Timedelta(hours=1)).isoformat()})
                record[stream].append({"opinion_id":f"{record['sample_id']}_{stream}_future",
                    "text":"Future market update must be excluded.",
                    "published_at":(as_of+pd.Timedelta(days=1)).isoformat(),
                    "available_at":(as_of+pd.Timedelta(days=1)).isoformat(),
                    "engagement":99999,"engagement_observed_at":(as_of+pd.Timedelta(days=1)).isoformat()})
            samples.append(record)
    return samples


def timestamp(value):
    time=pd.Timestamp(value)
    if pd.isna(time):raise ValueError("Missing timestamp")
    return time.tz_localize("UTC") if time.tzinfo is None else time.tz_convert("UTC")


def prepare(record,max_opinions=5):
    if max_opinions<1:raise ValueError("max_opinions must be positive")
    output={key:record[key] for key in ["sample_id","company_id","as_of"]}
    if "label" in record:
        if isinstance(record["label"],bool) or record["label"] not in [0,1,2]:raise ValueError("Invalid sentiment label")
        output["label"]=int(record["label"])
    cutoff=timestamp(record["as_of"])
    for stream in ["recency","popularity"]:
        eligible=[];seen=set()
        for opinion in record[stream]:
            if not isinstance(opinion["text"],str) or not opinion["text"].strip():raise ValueError("Empty opinion text")
            published=timestamp(opinion["published_at"]);available=timestamp(opinion["available_at"])
            if available<published:raise ValueError("Availability precedes publication")
            if published>cutoff or available>cutoff:continue
            if stream=="popularity":
                if timestamp(opinion["engagement_observed_at"])>cutoff:continue
                if not isinstance(opinion["engagement"],(int,float)) or not np.isfinite(opinion["engagement"]) or opinion["engagement"]<0:
                    raise ValueError("Invalid engagement")
            if opinion["opinion_id"] in seen:continue
            seen.add(opinion["opinion_id"]);eligible.append(dict(opinion))
        if stream=="recency":eligible.sort(key=lambda x:timestamp(x["published_at"]),reverse=True)
        else:eligible.sort(key=lambda x:(x["engagement"],timestamp(x["published_at"])),reverse=True)
        output[stream]=eligible[:max_opinions]
        if not output[stream]:raise ValueError(f"No eligible {stream} opinions at assessment time")
    return output


def partitions(samples,seed=31):
    groups=np.array([row["company_id"] for row in samples])
    dev,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(groups,groups=groups))
    train_rel,valid_rel=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed+1).split(dev,groups=groups[dev]))
    return {"train":dev[train_rel],"validation":dev[valid_rel],"test":test}


class Vocabulary:
    def __init__(self,mapping=None):self.mapping=mapping or {"[PAD]":0,"[UNK]":1}

    @staticmethod
    def tokenize(text):return re.findall(r"[a-z]+|[\u4e00-\u9fff]",text.lower())

    def fit(self,samples):
        counts=Counter(token for row in samples for stream in ["recency","popularity"]
                       for opinion in row[stream] for token in self.tokenize(opinion["text"]))
        for token,_ in sorted(counts.items(),key=lambda pair:(-pair[1],pair[0])):
            if token not in self.mapping:self.mapping[token]=len(self.mapping)
        return self

    def encode(self,text,max_tokens):
        return [self.mapping.get(word,1) for word in self.tokenize(text)[:max_tokens]] or [1]


class LocalBertTokenizer:
    def __init__(self,path):
        from transformers import AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        if self.tokenizer.pad_token_id!=0:raise ValueError("This adapter requires BERT padding token ID 0")

    def encode(self,text,max_tokens):
        return self.tokenizer.encode(text,truncation=True,max_length=max_tokens,add_special_tokens=True)


def collate(samples,tokenizer,max_tokens=32):
    result={}
    for stream in ["recency","popularity"]:
        encoded=[[tokenizer.encode(opinion["text"],max_tokens) for opinion in row[stream]] for row in samples]
        opinions=max(map(len,encoded));length=max(len(tokens) for row in encoded for tokens in row)
        ids=torch.zeros((len(samples),opinions,length),dtype=torch.long)
        for i,row in enumerate(encoded):
            for j,tokens in enumerate(row):ids[i,j,:len(tokens)]=torch.tensor(tokens)
        result[stream]=ids
    if all("label" in row for row in samples):result["labels"]=torch.tensor([row["label"] for row in samples])
    return result
