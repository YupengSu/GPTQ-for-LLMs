import torch
import torch.nn as nn
import time
from tqdm import tqdm

from .data import get_loaders 
from .quantize import Quantizer, make_quant3, Quant3Linear
from .gptq import GPTQ

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def prepare_calibration_input(model, dataloader, nsamples, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.encoder.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "rotary_pos_emb": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, hidden_states, attention_mask, rotary_pos_emb, **kwargs):
            inps[cache['i']] = hidden_states.squeeze(1)
            cache['i'] += 1
            cache['attention_mask'] = attention_mask
            cache['rotary_pos_emb'] = rotary_pos_emb
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    rotary_pos_emb= cache['rotary_pos_emb']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, rotary_pos_emb

@torch.no_grad()
def quant_gptq(args, model, tokenizer, device=torch.device("cuda:0")):
    ## GPTQ code available at: https://github.com/IST-DASLab/gptq/blob/main/llama.py
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, rotary_pos_emb= prepare_calibration_input(model, dataloader, args.nsamples, device)
    
    layers = model.transformer.encoder.layers
    start_time = time.time()
    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = GPTQ(subset[name])
            wrapped_layers[name].quantizer = Quantizer()
            wrapped_layers[name].quantizer.configure(
                args.wbits, perchannel=True, sym=args.sym, mse=False
            )

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
        
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(1), attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)[0].squeeze(1)
        for h in handles:
            h.remove()

        for name in subset:
            print(f"quantizing layer {i} name {name}")
            wrapped_layers[name].fasterquant(
                percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
            )
            quantizer = Quantizer()
            quantizer.scale = torch.zeros_like(wrapped_layers[name].quantizer.scale, device=device)
            quantizer.zero = torch.zeros_like(wrapped_layers[name].quantizer.scale, device=device)
            for name1 in wrapped_layers[name].groups:
                quantizer.scale = torch.cat((quantizer.scale, name1.scale), dim=1)
                quantizer.zero = torch.cat((quantizer.zero, name1.zero), dim=1)
            quantizers['model.layers.%d.%s' % (i, name)] = quantizer
            weight = subset[name].weight.data
            print(f"layer {name} scale: {quantizer.scale[:,1]} zero: {quantizer.zero[:,1]}")
            print(f"layer {name} weight: {weight[:,1]}")
            wrapped_layers[name].free()

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(1), attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)[0].squeeze(1)
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

    return quantizers

@torch.no_grad()
def quant_minmax(args, model, tokenizer, device=torch.device("cuda:0")):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.transformer.encoder.layers
    start_time = time.time()
    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            print(f"quantizing layer {i} name {name}")
            weight = subset[name].weight.data
            h, w = weight.shape
            _gs = args.groupsize
            qweight = torch.zeros(h, w).to(device)
            qscales = torch.zeros(h//_gs, w).to(device)
            for k in range(0, h, _gs):
                for j in range(w):
                    block = weight[k:k+_gs, j]
                    qblock, scale = quantize_group(block, args.wbits)
                    qweight[k:k+_gs, j] = qblock
                    qscales[k//_gs, j] = scale.item()
            subset[name].weight.data = qweight
            quantizer = Quantizer()
            quantizer.scale = qscales
            quantizer.zero = torch.zeros_like(qscales)
            quantizers['model.layers.%d.%s' % (i, name)] = quantizer
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

    return quantizers

def quantize_group(tensor, num_bits):
    if num_bits < 1:
        raise ValueError("num_bits must be a positive integer")
    quant_min = -2 ** (num_bits - 1) + 1
    quant_max = 2 ** (num_bits - 1) - 1
    tensor_min = torch.min(tensor)
    tensor_max = torch.max(tensor)
    scale = (tensor_max - tensor_min) / (quant_max - quant_min)
    tensor_normalized = (tensor - tensor_min) / (tensor_max - tensor_min)
    tensor_quantized = torch.round(tensor_normalized * (quant_max - quant_min) + quant_min)
    tensor_quantized = torch.clamp(tensor_quantized, quant_min, quant_max)

    return tensor_quantized, scale

def llm_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model