import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

sm = torch.nn.Softmax(dim=-1)

def get_topk(logits, k=5):
    pred = torch.topk(logits[0][-1], dim=-1, k=k)
    ret = list(zip([tokenizer.decode(x) for x in pred.indices.detach().tolist()], pred.values.detach().tolist()))
    return ret

def get_final_logits(sent=None, input_ids=None, k=5):
    """Get logit for a top k next tokens given previous context."""

    # form inputs
    if input_ids is None: inputs = tokenizer(sent, return_tensors="pt")
    else: inputs = {'input_ids': input_ids}

    # logits
    outputs = model(**inputs, labels=inputs["input_ids"], output_hidden_states=True, output_attentions=True)
    logits = sm(outputs.logits)

    # top-k
    ret = get_topk(logits, k=k)

    return ret, inputs, outputs

def logit_lens(outputs, k=5):
    """Return top-k predicts after each layer based on the logit lens idea
    The alignment forum is cringe but src: https://www.alignmentforum.org/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens"""
    ret = []
    for i in range(len(outputs.hidden_states)):
        ret.append(get_topk(sm(model.lm_head(outputs.hidden_states[i]))))
    return ret

def get_specific_logits(sent=None, input_ids=None, options: list=[]):
    """Get logit for a specified next token given previous context."""

    # form inputs
    if input_ids is None: inputs = tokenizer(sent, return_tensors="pt")
    else: inputs = {'input_ids': input_ids}

    # encode next token options
    options = [[tokenizer.encode(" " + x)[0]] if type(x) == str else [x] for x in options]

    # logits
    outputs = model(**inputs, labels=inputs["input_ids"], output_hidden_states=True, output_attentions=True)
    logits = sm(outputs.logits)
    ret = [(tokenizer.decode(x), float(logits[0][-1][x].detach())) for x in options]

    return ret, inputs, outputs

def get_top_attentions(fr, to, outputs):
    """Sort heads by attention on tokens fr->to in a sentence."""
    l = []
    for i in range(len(outputs.attentions)):
        for j in range(12):
            l.append((float(outputs.attentions[i][0][j][fr][to].detach().numpy()), i, j))
    l.sort()
    return l[::-1]

def draw_tensor(tensor: torch.Tensor, inputs):
    """Draw an attention pattern as a heatmap."""
    tokens = [tokenizer.decode(x).replace(' ', '_') for x in inputs['input_ids'][0]]
    tensor = tensor.detach()
    plt.imshow(tensor, cmap='bwr', interpolation='nearest', vmin=-1, vmax=1)
    plt.xticks(ticks=list(range(len(tokens))), labels=tokens, rotation='vertical')
    plt.yticks(ticks=list(range(len(tokens))), labels=tokens)
    plt.gca().xaxis.tick_top()
    plt.show()

def make_sents(template, vars):
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

def compare_attention_and_probs(sents, check, fr=-1, to=0, draw=[]):
    """Correlation between attention heads and output probs for some tokens"""
    ct = Counter()
    attns = []

    for sent in tqdm(s):

        # get token which we want to check last-state logits for
        var = check.split('.')
        if len(var) == 1: selected = sent[2][var[0]]
        else: selected = sent[2][var[0]][int(var[1])]

        # collect last logits, and top attentions for fr->to
        preds, inputs, outputs = get_specific_logits(input_ids=sent[0], options=[selected])
        for layer, head in draw:
            draw_tensor(outputs.attentions[layer][0][head], inputs)
        attentions = get_top_attentions(sent[1][fr] if fr >= 0 else fr, sent[1][to], outputs)
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