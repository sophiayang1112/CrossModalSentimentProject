import argparse
import copy
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix,ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import LABELS,simulate,prepare,partitions,Vocabulary,LocalBertTokenizer,collate
from .network import SentimentNetwork

VARIANTS=["full","no_cross","no_mfb","recency_only","popularity_only"]


def evaluate(labels,probabilities):
    predicted=probabilities.argmax(axis=1)
    return {"accuracy":float(accuracy_score(labels,predicted)),
        "weighted_f1":float(f1_score(labels,predicted,average="weighted",zero_division=0)),
        "macro_f1":float(f1_score(labels,predicted,average="macro",zero_division=0)),
        "weighted_precision":float(precision_score(labels,predicted,average="weighted",zero_division=0)),
        "weighted_recall":float(recall_score(labels,predicted,average="weighted",zero_division=0)),
        "confusion_matrix":confusion_matrix(labels,predicted,labels=[0,1,2]).tolist()}


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(2)


def batches(samples,tokenizer,batch_size,max_tokens,device,shuffle=False,seed=31):
    indices=np.random.default_rng(seed).permutation(len(samples)) if shuffle else np.arange(len(samples))
    for start in range(0,len(indices),batch_size):
        selected=[samples[i] for i in indices[start:start+batch_size]]
        batch=collate(selected,tokenizer,max_tokens)
        yield {key:values.to(device) for key,values in batch.items()}


def predict_probabilities(model,samples,tokenizer,batch_size,max_tokens,device):
    model.eval();output=[]
    with torch.no_grad():
        for batch in batches(samples,tokenizer,batch_size,max_tokens,device):
            output.append(model(batch["recency"],batch["popularity"]).softmax(-1).cpu().numpy())
    return np.concatenate(output)


def train(args):
    seed_everything(args.seed)
    if args.device=="cuda" and not torch.cuda.is_available():raise ValueError("CUDA device unavailable")
    if args.encoder=="bert" and not args.bert_path:raise ValueError("BERT mode requires --bert-path with local model files")
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    raw=json.loads(Path(args.data).read_text(encoding="utf-8")) if args.data else simulate(args.companies,args.seed)
    if len({row["sample_id"] for row in raw})!=len(raw):raise ValueError("Duplicate sample IDs")
    samples=[prepare(row,args.max_opinions) for row in raw]
    parts=partitions(samples,args.seed)
    datasets={name:[samples[i] for i in ids] for name,ids in parts.items()}
    labels={name:np.array([row["label"] for row in values]) for name,values in datasets.items()}
    for name,y in labels.items():
        if set(y)!={0,1,2}:raise ValueError(f"All sentiment classes required in {name}")
    tokenizer=LocalBertTokenizer(args.bert_path) if args.encoder=="bert" else Vocabulary().fit(datasets["train"])
    vocab_size=len(tokenizer.mapping) if args.encoder=="learned" else 0
    config={"vocab_size":vocab_size,"dim":args.dim,"heads":args.heads,"factors":args.factors,
        "max_opinions":args.max_opinions,"dropout":.1,"bert_path":args.bert_path if args.encoder=="bert" else None}
    distributions={name:np.bincount(y,minlength=3).tolist() for name,y in labels.items()}
    groups={name:set(row["company_id"] for row in values) for name,values in datasets.items()}
    assert not groups["train"]&groups["test"] and not groups["train"]&groups["validation"] and not groups["validation"]&groups["test"]
    majority=int(np.bincount(labels["train"],minlength=3).argmax())
    majority_p=np.zeros((len(labels["test"]),3));majority_p[:,majority]=1
    results={"majority_baseline":evaluate(labels["test"],majority_p)}
    histories={};best_full=None;full_probabilities=None
    weight=len(labels["train"])/(3*np.bincount(labels["train"],minlength=3))
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weight,dtype=torch.float32,device=args.device))
    for variant in VARIANTS:
        seed_everything(args.seed)
        model=SentimentNetwork(**config,variant=variant).to(args.device)
        optimizer=torch.optim.Adam(model.parameters(),lr=args.learning_rate)
        best=-1.;state=None;history=[];stale=0
        for epoch in range(args.epochs):
            model.train();loss_sum=0.;count=0
            for batch in batches(datasets["train"],tokenizer,args.batch_size,args.max_tokens,args.device,True,args.seed+epoch):
                optimizer.zero_grad(set_to_none=True)
                logits=model(batch["recency"],batch["popularity"])
                loss=criterion(logits,batch["labels"])
                if not torch.isfinite(loss):raise RuntimeError("Nonfinite training loss")
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step();loss_sum+=float(loss.detach())*len(logits);count+=len(logits)
            valid=predict_probabilities(model,datasets["validation"],tokenizer,args.batch_size,args.max_tokens,args.device)
            validation=evaluate(labels["validation"],valid)
            history.append({"epoch":epoch+1,"train_loss":loss_sum/count,"validation_macro_f1":validation["macro_f1"]})
            if validation["macro_f1"]>best+1e-8:
                best=validation["macro_f1"];state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()};stale=0
            else:stale+=1
            if stale>=args.patience:break
        model.load_state_dict(state);model.to(args.device)
        started=time.perf_counter()
        probabilities=predict_probabilities(model,datasets["test"],tokenizer,args.batch_size,args.max_tokens,args.device)
        if args.device=="cuda":torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        results[variant]=evaluate(labels["test"],probabilities)
        results[variant]["inference_ms_per_sample"]=1000*elapsed/len(probabilities)
        results[variant]["best_validation_macro_f1"]=best
        results[variant]["epochs_run"]=len(history)
        results[variant]["trainable_parameters"]=sum(p.numel() for p in model.parameters() if p.requires_grad)
        histories[variant]=history
        print(f"{variant}: accuracy={results[variant]['accuracy']:.3f}; macro_f1={results[variant]['macro_f1']:.3f}",flush=True)
        if variant=="full":best_full=model;full_probabilities=probabilities
    torch.save({"state_dict":{key:value.cpu() for key,value in best_full.state_dict().items()},"config":config,
        "vocabulary":tokenizer.mapping if args.encoder=="learned" else None,"encoder":args.encoder,
        "max_tokens":args.max_tokens,"labels":LABELS},output/"model.pt")
    example=datasets["test"][0];best_full.to(args.device);best_full.eval()
    batch=collate([example],tokenizer,args.max_tokens)
    with torch.no_grad():
        _,weights=best_full(batch["recency"].to(args.device),batch["popularity"].to(args.device),return_attention=True)
    attention={key:value[0].mean(0).cpu().numpy() for key,value in weights.items()}
    attention_report={"sample_id":example["sample_id"],"company_id":example["company_id"],
        "as_of":example["as_of"],"recency_opinion_ids":["CLS"]+[op["opinion_id"] for op in example["recency"]],
        "popularity_opinion_ids":["CLS"]+[op["opinion_id"] for op in example["popularity"]],
        "stage_one":attention["trending_to_timely"].tolist(),"stage_two":attention["attended_to_trending"].tolist(),
        "interpretation":"Attention weights describe model routing, not causal explanations."}
    report={"data_source":"user-supplied JSON" if args.data else "synthetic financial opinion streams",
        "seed":args.seed,"samples":len(samples),"companies":len(set(row["company_id"] for row in samples)),
        "encoder":args.encoder,"pretrained_bert_executed":args.encoder=="bert",
        "architecture":{key:value for key,value in config.items() if key!="bert_path"},
        "versions":{"python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__},
        "training":{"epochs_limit":args.epochs,"learning_rate":args.learning_rate,"batch_size":args.batch_size,
                    "patience":args.patience,"optimizer":"Adam","device":args.device},
        "partition_sizes":{name:len(values) for name,values in datasets.items()},"class_counts":distributions,
        "company_overlap":0,"selection":"checkpoint chosen by validation macro-F1 within each variant",
        "holdout_metrics":results,"histories":histories}
    (output/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (output/"attention_example.json").write_text(json.dumps(attention_report,indent=2),encoding="utf-8")
    (output/"scoring_inputs.json").write_text(json.dumps([{key:value for key,value in row.items() if key!="label"}
        for row in datasets["test"][:8]],indent=2),encoding="utf-8")
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for variant,history in histories.items():axes[0].plot([h["epoch"] for h in history],[h["validation_macro_f1"] for h in history],label=variant)
    axes[0].set(xlabel="Epoch",ylabel="Validation macro-F1",title="Learning curves");axes[0].legend(fontsize=8)
    names=list(results);position=np.arange(len(names))
    axes[1].bar(position-.18,[results[n]["accuracy"] for n in names],.36,label="accuracy")
    axes[1].bar(position+.18,[results[n]["weighted_f1"] for n in names],.36,label="weighted F1")
    axes[1].set_xticks(position,names,rotation=35,ha="right");axes[1].legend();axes[1].set_title("Company-disjoint holdout");axes[1].set_ylim(0,1)
    fig.suptitle(f"Financial opinion integration | {args.encoder} opinion encoder")
    fig.tight_layout();fig.savefig(output/"model_comparison.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.5,4.5))
    ConfusionMatrixDisplay.from_predictions(labels["test"],full_probabilities.argmax(1),labels=[0,1,2],display_labels=LABELS,colorbar=False,ax=ax)
    ax.set_title("Full model | sentiment classification")
    fig.tight_layout();fig.savefig(output/"confusion_matrix.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,title in zip(axes,["trending_to_timely","attended_to_trending"],["Stage 1: trending queries timely","Stage 2: attended queries trending"]):
        image=ax.imshow(attention[key],aspect="auto",vmin=0,vmax=1,cmap="Blues")
        ax.set_title(title);ax.set_xlabel("Key opinion index (0 = CLS)");ax.set_ylabel("Query opinion index (0 = CLS)")
        fig.colorbar(image,ax=ax)
    fig.tight_layout();fig.savefig(output/"cross_attention.png",dpi=150);plt.close(fig)
    return report


def score(model_path,input_path,output_path):
    torch.set_num_threads(2)
    checkpoint=torch.load(model_path,map_location="cpu",weights_only=True)
    tokenizer=Vocabulary(checkpoint["vocabulary"]) if checkpoint["encoder"]=="learned" else LocalBertTokenizer(checkpoint["config"]["bert_path"])
    samples=[prepare(row,checkpoint["config"]["max_opinions"]) for row in json.loads(Path(input_path).read_text())]
    model=SentimentNetwork(**checkpoint["config"]);model.load_state_dict(checkpoint["state_dict"])
    probabilities=predict_probabilities(model,samples,tokenizer,32,checkpoint["max_tokens"],"cpu")
    result=[{"sample_id":row["sample_id"],"sentiment":LABELS[int(p.argmax())],
             "probabilities":dict(zip(LABELS,map(float,p)))} for row,p in zip(samples,probabilities)]
    Path(output_path).write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"Saved {len(result)} sentiment predictions to {output_path}")


def main():
    parser=argparse.ArgumentParser(description="Cross-attention financial sentiment integration")
    sub=parser.add_subparsers(dest="command",required=True)
    fit=sub.add_parser("train")
    for name,default in [("companies",600),("seed",31),("epochs",50),("patience",8),("batch-size",32),
                         ("max-tokens",32),("max-opinions",5),("dim",64),("heads",4),("factors",4)]:
        fit.add_argument('--'+name,type=int,default=default)
    fit.add_argument("--learning-rate",type=float,default=.002)
    fit.add_argument("--encoder",choices=["learned","bert"],default="learned");fit.add_argument("--bert-path")
    fit.add_argument("--device",choices=["cpu","cuda"],default="cpu");fit.add_argument("--output",default="reports");fit.add_argument("--data")
    predict=sub.add_parser("score");predict.add_argument("--model",default="reports/model.pt")
    predict.add_argument("--input",default="reports/scoring_inputs.json");predict.add_argument("--output",default="predictions.json")
    args=parser.parse_args()
    if args.command=="train":
        if min(args.epochs,args.patience,args.batch_size,args.max_tokens,args.max_opinions,args.heads,args.factors)<1:
            parser.error("Training sizes must be positive")
        if args.learning_rate<=0:parser.error("Learning rate must be positive")
        train(args)
    else:score(args.model,args.input,args.output)


if __name__=="__main__":main()
