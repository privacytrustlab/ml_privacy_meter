## Edited and reused from  https://github.com/nyu-dl/bert-gen/blob/master/bert-babble.ipynb

import os
import random

import numpy as np
from numpy.core.fromnumeric import shape
import torch
from transformers import BertForMaskedLM, BertTokenizer
from torch.distributions.categorical import Categorical
from datetime import datetime
import random
import json
import pandas as pd
import csv 
import argparse
import random
import torch.nn.functional as F


random.seed(5)

################

parser = argparse.ArgumentParser(description="Sample extraction")

parser.add_argument("--max_iter", type=int, help="number of changes to make in the gibbs chain", default=100)
parser.add_argument("--n_samples", type=int, help="number of changes to make in the gibbs chain", default=50)
parser.add_argument("--batch_size", type=int, help="number of changes to make in the gibbs chain", default=1)
parser.add_argument("--max_len", type=int, help="number of changes to make in the gibbs chain", default=10)
parser.add_argument("--min_len", type=int, help="number of changes to make in the gibbs chain", default=5)
parser.add_argument("--rand_len", type=bool, help="number of changes to make in the gibbs chain", default=False)


parser.add_argument("--temperature", type=float, help="number of changes to make in the gibbs chain", default=1.0)
parser.add_argument("--gamma", type=float, help="number of changes to make in the gibbs chain", default=1.0)
parser.add_argument("--theta", type=int, help="number of changes to make in the gibbs chain", default=0)

parser.add_argument("--incl_full_name", type=float, help="number of changes to make in the gibbs chain", default=0.0)
parser.add_argument("--incl_any_name", type=float, help="number of changes to make in the gibbs chain", default=0.0)
parser.add_argument("--incl_dis", type=float, help="number of changes to make in the gibbs chain", default=0.0)


parser.add_argument("--chkp", type=bool, help="number of changes to make in the gibbs chain", default=False)



parser.add_argument("--block", action='store_true')
parser.add_argument("--no_block", type=int, default=5)
parser.add_argument("--contigous", action='store_true')



parser.add_argument("--shuffle_positions", action='store_true')
parser.add_argument("--anneal_gamma", action='store_true')
parser.add_argument("--anneal_temp", action='store_true')



###degenerate gibbs sampler
parser.add_argument("--top_k", type=int, help="top_k sampler-so far only degenerate support", default=0)
parser.add_argument("--burnin", type=int, help="burn in for degenerate support", default=250)



parser.add_argument("--out_path", type=str, help="dir", default='./batched')
parser.add_argument("--input_file", type=str, help="dir", default='smaller_samples_len_22.csv')
parser.add_argument("--model_name", type=str, help="dir", default='clinicalbert_1a')
parser.add_argument("--model_path", type=str, help="dir", default='MODEL_PATH')
parser.add_argument("--tok_path", type=str, help="dir", default='bert-base-uncased')



parser.add_argument('--seed',type=str, default="warm")
parser.add_argument('--no_freeze', action='store_true')


args = parser.parse_args()

##################

cuda = torch.cuda.is_available()
print(cuda)
device = 'cuda' if cuda else 'cpu'

# Load pre-trained model (weights)
model_version = args.model_path #os.environ["MODEL_PATH"]
model = BertForMaskedLM.from_pretrained(model_version)
model.eval()

if cuda:
    model = model.cuda()

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(args.tok_path, do_lower_case=True)




def sent_list(csv_file):
    sent_lists = []
    with open(csv_file, 'r') as file_ext:
        csv_reader = csv.reader(file_ext)
        header = next(csv_reader)
        
        for row in csv_reader:
            
            sent_lists.append(row[3])
            
    return sent_lists 




def tokenize_batch(batch):
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]


def untokenize_batch(batch):
    return [tokenizer.convert_ids_to_tokens(list(sent.to('cpu').numpy())) for sent in batch]


def detokenize(sent):
    """ Roughly detokenizes (mainly undoes wordpiece) """
    new_sent = []
    for i, tok in enumerate(sent):
        if tok.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
        else:
            new_sent.append(tok)
    return new_sent


CLS = "[CLS]"
SEP = "[SEP]"
MASK = "[MASK]"
mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
sep_id = tokenizer.convert_tokens_to_ids([SEP])[0]
cls_id = tokenizer.convert_tokens_to_ids([CLS])[0]
mr_id = 2720 #tokenizer.convert_tokens_to_ids("mr")[0]
ms_id = 5796 #tokenizer.convert_tokens_to_ids("ms")[0]




def generate_step(out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True):
    """Generate a word from from out[gen_idx]

    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Overridden by top_k
    """
    logits = out[:, gen_idx]
    if temperature is not None:
        logits = logits / temperature
    if top_k > 0:
        kth_vals, kth_idx = logits.topk(top_k, dim=-1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
    elif sample:
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample().squeeze(-1)
    else:
        idx = torch.argmax(logits, dim=-1)
    return idx.tolist() if return_list else idx


def get_init_text(seed_text, max_len, batch_size=1, rand_init=False):
    """ Get initial sentence by padding seed_text with either masks or random words to max_len """
    batch = [seed_text +  [SEP]  for _ in range(batch_size)] #TODO

    return tokenize_batch(batch)


def printer(sent, should_detokenize=True):
    if should_detokenize:
        sent = detokenize(sent)[1:-1]
    print(" ".join(sent))


def to_file(sents, file):
    with open(file, "a") as f:
        f.write("\n".join(sents) + "\n")


# Generation modes as functions
import math
import time



#   return  score
  
def energy_score(batch):
  #encoded_input = tokenizer(text, return_tensors='pt').to(device)
  #tokens = encoded_input['input_ids'][0]
  seq_len = len(batch[0])-2
  posns = [i+1 for i in range(seq_len)]
  #random.shuffle(posns)
  norm_score = [0.0] * batch.shape[0]
  raw_score = [0.0] * batch.shape[0]
  for posn in posns:
    old_wrd = batch[:,posn].clone()
    #print(tokenizer.decode(tokens[1:-1]))
    batch[:,posn] = mask_id
    #output = model(**encoded_input)[0][0,posn,:].log_softmax(dim=-1)
    #output = model(**encoded_input)[0][0,posn,:]
    output = model(batch)[:,posn,:]
    norm_output = output.log_softmax(dim=-1)
    for i in range(batch.shape[0]): #TODO check this
        raw_score[i] += output[i,old_wrd[i]].item()
        norm_score[i] += norm_output[i,old_wrd[i]].item()
    #raw_score += output[old_wrd].item()
    #norm_score += norm_output[old_wrd].item()
    batch[:,posn] = old_wrd
  return [-1.0*raw_s for raw_s in raw_score], [-1.0*norm_s for norm_s in norm_score]

def parallel_sequential_generation(
    seed_text,
    file_list,
    batch_size=10,
    max_len=15,
    top_k=0,
    temperature=1,
    max_iter=300,
    burnin=200,
    cuda=False,
    print_every=10,
    verbose=True,
    args=args,
    chkp_list=[10,25,50,100]
):
    """Generate for one random position at a timestep

    args:
        - burnin: during burn-in period, sample from full distribution; afterwards take argmax
    """
    #     seed_len = len(seed_text)
    
    losses = []
 

    
    

    
    
    posns =  [i for i, y in enumerate(seed_text)  if (y != cls_id and y!= sep_id)] 
    #nmasks = int(mlm_prob*(len(seed_text)-1))       #-1 because of cls
    batch = torch.tensor(get_init_text(seed_text, max_len, batch_size)).to(device)
    labels = batch.clone()
    
    
    
    for iter in posns:
        batch = torch.tensor(get_init_text(seed_text, max_len, batch_size)).to(device)
   
        labels = batch.clone()    
        mask_pos =iter

        #seed_text[mask_pos] = MASK
        batch[:,mask_pos] = mask_id

        labels[batch != mask_id] = -100 

        logits = model(batch,labels=labels)['logits']
        loss = model(batch,labels=labels)['loss']
      

        loss2 = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), labels.view(-1))
        
        assert loss==loss2
        
        losses.append(loss2.item())


    return untokenize_batch(batch),[],sum(losses)/len(losses)


def generate(
    n_samples,
    file_list,
    seed_text="[CLS]",
    batch_size=10,
    max_len=25,
    top_k=100,
    temperature=1.0,
    burnin=200,
    max_iter=500,
    cuda=False,
    print_every=1,
    args=args,
    chkp_list=[10,25,50,100]
    
):
    # main generation function to call
    sentences = []
    n_batches = math.ceil(n_samples / batch_size)
    start_time = time.time()
    if args.rand_len:
        max_len = random.randint(args.min_len,args.max_len)
    for batch_n in range(n_batches):
        batch , metadata, old_r = parallel_sequential_generation(
            seed_text,
            file_list=file_list,
            batch_size=batch_size,
            max_len=max_len,
            top_k=top_k,
            temperature=temperature,
            burnin=burnin,
            max_iter=max_iter,
            cuda=cuda,
            verbose=False,
            chkp_list=chkp_list
        )

        if (batch_n + 1) % print_every == 0:
            #print("Finished batch %d in %.3fs" % (batch_n + 1, time.time() - start_time))
            start_time = time.time()

        sentences += batch
    return sentences, metadata, old_r




#
top_k = args.top_k #40 #not used
#leed_out_len = 5  # max_len, not used
burnin = args.burnin #250 #not used
temperature = args.temperature
###########

dirname = args.out_path
n_samples = args.n_samples
batch_size = args.batch_size
max_len =  args.max_len
max_iter = args.max_iter
seed = args.seed


seed_name = seed

now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

folder_name = "get_sentence_loss_file_enorm_{}_model_{}_nsamples_{}_date_{}".format(args.input_file,args.model_name, args.n_samples,dt_string)

directory = "{}/{}".format(dirname,folder_name)
if not os.path.exists(directory):
    os.mkdir(directory)

dirname=directory




if args.chkp:
    chkp_list=[10,25,50,100,200,300,400,500,600,700,800,900,1000]
    file_list=[]
    for element in chkp_list:
        file_list.append(open(f'{dirname}/samples_{element}.txt','w+'))
        print(element)
else:
    chkp_list=[]
    file_list=[]
    


input_file = './CSV_Files/'+args.input_file 
sents_list = sent_list(input_file)

with open(f"{dirname}/samples.txt", "a") as f ,open(f"{dirname}/metadata.txt", "a") as f_meta, open(f"{dirname}/loss.txt", "a") as f_energy:
    for i in range(len(sents_list)): #was 200
        

        if seed == "fresh" :
            seed_text = "[CLS]"
        elif seed == 'warm':
            seed_text = '[CLS]'+' '+ sents_list[i]+' ' 
        else:
            seed_text = seed

        print(seed_text)
        seed_text = tokenizer.tokenize(seed_text)
        #print(seed_text)
        print(len(seed_text))
        #seed_text = seed_text.split()

        
        torch.cuda.empty_cache()
        bert_sents, meta_data,old_r = generate(
            n_samples,
            file_list=file_list,
            seed_text=seed_text,
            batch_size=batch_size,
            max_len=max_len,
            top_k=top_k,
            temperature=temperature,
            burnin=burnin,
            max_iter=max_iter,
            cuda=cuda,
            args=args,
            chkp_list = chkp_list
        )
        print(old_r)
        
        f_energy.write(str(old_r)+"\n")
        f_energy.flush()
        
        sents = list(map(lambda x: " ".join(detokenize(x)), bert_sents))
        f.write("\n".join(sents) + "\n")
        f.flush()
        meta_data_str = [str(l) for l in meta_data]
        f_meta.write("\n".join(meta_data_str)+"\n")
        f_meta.flush
