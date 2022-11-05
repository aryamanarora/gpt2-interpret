import torch
from transformers import AutoTokenizer, AutoModel, GPTNeoXForCausalLM, GPT2LMHeadModel
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

sm = torch.nn.Softmax(dim=-1)
cos = torch.nn.CosineSimilarity(eps=1e-6, dim=-1)

def get_embed_in(model):
    """Get input embedding layer weight matrix."""
    if 'pythia' in model.config._name_or_path and type(model) == GPTNeoXForCausalLM:
        return model.gpt_neox.embed_in.weight
    elif 'gpt2' in model.config._name_or_path and type(model) == GPT2LMHeadModel:
        return model.transformer.wte.weight

def get_embed_out(model):
    """Get output embedding layer weight matrix."""
    if 'pythia' in model.config._name_or_path and type(model) == GPTNeoXForCausalLM:
        return model.embed_out.weight
    elif 'gpt2' in model.config._name_or_path and type(model) == GPT2LMHeadModel:
        return model.transformer.wte.weight.t()

def clean(tokenizer, t):
    """Clean up token representation"""
    string = tokenizer.decode(t)
    string = string.replace('\n', '\\n')
    string = string.replace('_', '\\_')
    string = string.replace(' ', '_')
    return string

def get_topk(model, tokenizer, logits, k=5):
    pred = torch.topk(logits, dim=-1, k=k)
    ret = list(zip(pred.indices.detach().tolist(), pred.values.detach().tolist()))
    return ret

def analyse_embed_matrix(model, tokenizer, use_transpose=False):
    """Get norms of W_E, probabilties along the diagonal for W_E * W_U, and argmaxes for W_E * W_U"""
    norms, probs, maxes, probs_raw = [], [], [], []
    decoder = get_embed_in(model).t() if use_transpose else get_embed_out(model)

    # do in batches to save memory (otherwise we have to deal w a vocab * vocab matrix which is huge)
    for i in tqdm(range(0, get_embed_in(model).size()[0], 1000)):
        emb = torch.matmul(get_embed_in(model)[i:i+1000], decoder)
        unemb = torch.softmax(emb, dim=-1)
        norm = torch.norm(unemb, dim=-1)
        max = torch.argmax(unemb, dim=-1)
        norms.extend(norm.tolist())
        probs.extend(torch.diagonal(unemb[:,i:i+1000]).tolist())
        maxes.extend(max.tolist())
        probs_raw.extend(torch.diagonal(emb[:,i:i+1000]).tolist())
        del emb
        del unemb
        del norm

    return norms, probs, maxes, probs_raw

def get_closest_static_embeds(model, tokenizer, t, use_transpose=False, multiplier=1.0, k=20):
    """Get top probabilities for softmax(W_E[t] * W_U)"""
    decoder = get_embed_in(model).t() if use_transpose else get_embed_out(model)
    dot = torch.matmul(get_embed_in(model)[t] * multiplier, decoder)
    a = torch.softmax(dot, dim=-1)
    top = torch.topk(a, k=k)
    return [{
            'tok': index,
            'prob': float(val.detach()),
            'dot': torch.dot(get_embed_in(model)[t], decoder.t()[index]),
            'cos': cos(get_embed_in(model)[t], decoder.t()[index]),
            'norm': torch.norm(decoder.t()[index]),
        } for index, val in zip(top.indices, top.values)]

def get_all_logits(model, tokenizer, sent=None, input_ids=None, k=5):
    """Get model outputs and top logits for all tokens in input."""

    # form inputs
    if input_ids is None: inputs = tokenizer(sent, return_tensors="pt")
    else: inputs = {'input_ids': torch.tensor(input_ids)}
    if 'token_type_ids' in inputs: del inputs['token_type_ids']

    # logits
    outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
    logits = sm(outputs.logits)

    # top-k, all tokens
    ret = []
    for i in range(logits.size()[1]):
        ret.append(get_topk(model, tokenizer, logits[0][i], k=k))
    
    return ret, inputs, outputs

def get_final_logits(model, tokenizer, sent=None, input_ids=None, k=5):
    """Get logit for a top k next tokens given previous context."""

    # form inputs
    if input_ids is None: inputs = tokenizer(sent, return_tensors="pt")
    else: inputs = {'input_ids': input_ids}
    if 'token_type_ids' in inputs: del inputs['token_type_ids']

    # logits
    outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
    logits = sm(outputs.logits)

    # top-k, last token
    ret = get_topk(model, tokenizer, logits[0][-1], k=k)

    return ret, inputs, outputs

def get_specific_logits(model, tokenizer, sent=None, input_ids=None, options: list=[]):
    """Get logit for a specified next token given previous context."""

    # form inputs
    if input_ids is None: inputs = tokenizer(sent, return_tensors="pt")
    else: inputs = {'input_ids': input_ids}
    if 'token_type_ids' in inputs: del inputs['token_type_ids']

    # encode next token options
    options = [[tokenizer.encode(" " + x)[0]] if type(x) == str else [x] for x in options]

    # logits
    outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
    logits = sm(outputs.logits)
    ret = [(tokenizer.decode(x), float(logits[0][-1][x].detach())) for x in options]

    return ret, inputs, outputs

def logit_lens(model, tokenizer, outputs, k=5):
    """Return top-k predicts after each layer based on the logit lens idea
    The alignment forum is cringe but src: https://www.alignmentforum.org/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens"""
    ret = []
    for j in range(outputs.hidden_states[0].size()[1]):
        t = []
        for i in range(len(outputs.hidden_states)):
            t.append(get_topk(model, tokenizer, sm(torch.matmul(get_embed_out(model), outputs.hidden_states[i][0][j]))))
        ret.append(t)
    return ret

def get_top_attentions(model, tokenizer, fr, to, outputs):
    """Sort heads by attention on tokens fr->to in a sentence."""
    l = []
    heads = outputs.attentions[0][0].size()[0]
    for i in range(len(outputs.attentions)):
        for j in range(heads):
            l.append((float(outputs.attentions[i][0][j][fr][to].detach().numpy()), i, j))
    l.sort()
    return l[::-1]

def draw_tensor(model, tokenizer, tensor: torch.Tensor, inputs=None):
    """Draw an attention pattern as a heatmap."""
    tensor = tensor.detach()
    plt.imshow(tensor, cmap='bwr', interpolation='nearest', vmin=-1, vmax=1)
    if inputs:
        tokens = [tokenizer.decode(x).replace(' ', '_') for x in inputs['input_ids'][0]]
        plt.xticks(ticks=list(range(len(tokens))), labels=tokens, rotation='vertical')
        plt.yticks(ticks=list(range(len(tokens))), labels=tokens)
    plt.gca().xaxis.tick_top()
    plt.show()

def get_qk_ov(model, tokenizer, head: tuple[int]):
    layer, head = head[0], head[1]

    # get attention block for layer
    attn = model.transformer.h[layer].attn
    query, key, value = attn.c_attn.weight.split(attn.split_size, dim=1)

    # decompose into heads
    query = query.reshape(attn.split_size, attn.num_heads, attn.head_dim)[:,head,:].reshape(attn.split_size)
    key = key.reshape(attn.split_size, attn.num_heads, attn.head_dim)[:,head,:].reshape(attn.split_size)
    value = value.reshape(attn.split_size, attn.num_heads, attn.head_dim)[:,head,:].reshape(attn.split_size)
    output = attn.c_proj.weight


def make_sents(model, tokenizer, template, vars):
    """Generate sentences from templates."""
    template = template.replace('<', '><')
    template = [x.rstrip() for x in template.split('>') if x.rstrip()]

    ct = 1
    for i in vars: ct *= len(vars[i])
    sents = []

    for i in range(ct):
        s = []

        selected_vars = {}
        for var in vars:
            select = i % len(vars[var])
            selected_vars[var] = vars[var][select]
            i //= len(vars[var])

        l = 0
        poses = []
        for j in range(len(template)):
            c = template[j]

            if (c[0] == '<'):
                poses.append(l)
                var = c[1:].split('.')
                if len(var) == 1: selected = selected_vars[var[0]]
                else: selected = selected_vars[var[0]][int(var[1])]
                c = " " + selected if j != 0 else selected

            tok = tokenizer.encode(c)
            s.extend(tok)
            l += len(tok)
        
        sents.append((torch.tensor([s]), poses, selected_vars))
    return sents

def compare_attention_and_probs(model, tokenizer, sents, check, fr=-1, to=0, draw=[]):
    """Correlation between attention heads and output probs for some tokens"""
    ct = Counter()
    attns = []

    for sent in tqdm(sents):

        # get token which we want to check last-state logits for
        var = check.split('.')
        if len(var) == 1: selected = sent[2][var[0]]
        else: selected = sent[2][var[0]][int(var[1])]

        # collect last logits, and top attentions for fr->to
        preds, inputs, outputs = get_specific_logits(model, tokenizer, input_ids=sent[0], options=[selected])
        for layer, head in draw:
            draw_tensor(model, tokenizer, outputs.attentions[layer][0][head], inputs)
        attentions = get_top_attentions(model, tokenizer, sent[1][fr] if fr >= 0 else fr, sent[1][to], outputs)
        prob = preds[0][1]

        # overall max attentions
        subct = Counter()
        for i in attentions:
            ct[i[1:]] += i[0]
            subct[i[1:]] += i[0]
        attns.append([subct, prob])

    print(ct.most_common(10))
    return attns

def graph(attns, head):
    """Graph attention vs. output prob."""
    # attentions vs. probs
    X = [x[0][head] for x in attns]
    Y = [x[1] for x in attns]
    plt.scatter(X, Y)
    plt.xlabel('Attention activation')
    plt.ylabel('Output probability')

    # linear fit
    z = np.polyfit(X, Y, 1)
    print(z)
    p = np.poly1d(z)
    plt.plot(X,p(X),"r-")

    plt.show()