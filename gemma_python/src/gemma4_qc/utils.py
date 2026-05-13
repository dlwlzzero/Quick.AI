import os 
from datetime import datetime
import collections
import torch
import numpy as np
import json
import pickle
import itertools

input_keys_for_causal_llm = ['input_ids', 'past_key_values', 'attention_mask', 'position_ids', ]

model2config_lut = {
    'GPT2LMHeadModel'  : ('n_layer', 'n_head', 'n_positions', 'n_embd'),
    'GPTNeoForCausalLM': ('num_layers', 'num_heads', 'max_position_embeddings', 'hidden_size'),
    'LlamaForCausalLM': ('num_hidden_layers', 'num_attention_heads', 'max_position_embeddings', 'hidden_size'),
    'GaussForCausalLM': ('num_hidden_layers', 'num_attention_heads', 'max_position_embeddings', 'hidden_size'),
}
model2config_input_keys = {
    'GPT2LMHeadModel' : ('input_ids', 'past_key_values', 'attention_mask', 'position_ids'),
    'GPTNeoForCausalLM': ('input_ids', 'past_key_values', 'attention_mask', 'position_ids'),
    'LlamaForCausalLM': ('input_ids', 'attention_mask', 'position_ids', 'q_lora_A_weights','q_lora_B_weights', 'k_lora_A_weights','k_lora_B_weights','v_lora_A_weights','v_lora_B_weights', 'o_lora_A_weights','o_lora_B_weights', 'gate_lora_A_weights','gate_lora_B_weights', 'up_lora_A_weights','up_lora_B_weights', 'down_lora_A_weights','down_lora_B_weights','past_key_values'),
}


lora_supported_layers_list_global = []
lora_rank = None

def set_lora_supported_layers_list_and_rank(lora_supported_layers,rank):
    global lora_supported_layers_list_global
    global lora_rank
    lora_supported_layers_list_global = lora_supported_layers
    lora_rank = rank


def extract_info_from_model_cfg(model_cfg):
    return [getattr(model_cfg, key) for key in model2config_lut[model_cfg.architectures[0]]]

def get_model_inputs(model_cfg, test_vector):
    input_keys = model2config_input_keys[model_cfg.architectures[0]] 
    return {key: test_vector[key] for key in input_keys if key in test_vector}

def unify_kv_encodings(quant_sim, model_cfg):
    class Module:
        def __init__(self, format, is_input, index, keep_bitwidth=False):
            self.format, self.is_input, self.index, self.keep_bitwidth = format, is_input, index, keep_bitwidth
        def __repr__(self):
            return f'Module({self.foamt}, {self.is_input}, {self.index}, {self.keep_bitwidth})'

    def _copy(dst, src, keep_dst_bitwidth, factor=256):
        dst.encoding.min = src.encoding.min
        dst.encoding.max = src.encoding.max
        if keep_dst_bitwidth and dst.encoding.bw != src.encoding.bw:
            if dst.encoding.bw == 16:
                dst.encoding.offset = src.encoding.offset * factor
                dst.encoding.delta = src.encoding.delta / factor
            else:
                dst.encoding.offset = src.encoding.offset / factor
                dst.encoding.delta = src.encoding.delta * factor
        else:
            dst.encoding.bw = src.encoding.bw
            dst.encoding.offset = src.encoding.offset
            dst.encoding.delta = src.encoding.delta
        dst.use_symmetric_encodings = src.use_symmetric_encodings

    def _get_quantizers(mod, layer, head):
        name = _name(mod, layer, head)
        # print("_get_quantizers name",name)
        quantizers = modules[name].input_quantizers if mod.is_input else modules[name].output_quantizers
        return quantizers if mod.index < 0 else [quantizers[mod.index]]

    def _name(mod, layer, head):
        return mod.format.format(layer, head)

    from aimet_torch.qc_quantize_op import QcQuantizeWrapper
    assert model_cfg.architectures[0] == 'LlamaForCausalLM', 'Unsupported model: {model_cfg.architectures[0]}'

    Settings = {
        Module('model.layers.{}.self_attn.mm_qk.{}', True, 1): [            # 2nd input(K) of QK mamtul has same range as
            Module('model.layers.{}.self_attn.k_cat.{}', True, 0),         # inputs of concat(past_k, k)
            Module('model.layers.{}.self_attn.k_cat.{}', True, 1),
            Module('model.layers.{}.self_attn.k_rope.{}.stack', False, 0),  # output of rope
            Module('model.layers.{}.self_attn.k_rope.{}.add', False, 0),    # outputs of add of rope
            Module('model.layers.{}.self_attn.k_rope.{}.sub', False, 0),    # inputs of sub of rope
        ],
        Module('model.layers.{}.self_attn.mm_qk.{}', True, 0): [            # 2nd input(K) of QK mamtul has same range as
            Module('model.layers.{}.self_attn.q_rope.{}.stack', False, 0),  # output of rope
            Module('model.layers.{}.self_attn.q_rope.{}.add', False, 0),    # outputs of add of rope
            Module('model.layers.{}.self_attn.q_rope.{}.sub', False, 0),    # inputs of sub of rope
        ],
        Module('model.layers.{}.self_attn.mm_qkv.{}', True, 1): [           # 2nd input(V) of QKV matmul has same range as
            Module('model.layers.{}.self_attn.v_cat.{}', True, 0),          # inputs of concat(past_v, v)
            Module('model.layers.{}.self_attn.v_cat.{}', True, 1), 
            Module('model.layers.{}.self_attn.v_proj_sha.{}', False, 0, keep_bitwidth=True),    # output of v_proj
        ],
        Module('model.layers.{}.self_attn.o_cat', False, 0): [              # concatenated attn_output has same range as
            Module('model.layers.{}.self_attn.mm_qkv.{}', False, 0),        # inputs of concat(past_v, v)
        ],
    }

    errors = []
    num_layers, num_heads, *_ = extract_info_from_model_cfg(model_cfg)
    modules = {name:module for name, module in quant_sim.model.named_modules() if isinstance(module, QcQuantizeWrapper)}

    itr_cnt = 0
    for srcmod, dstmods in Settings.items():
        itr_cnt+=1
        for layer, head in itertools.product(range(num_layers), range(model_cfg.num_attention_heads)):
            # print("layer, head",layer, head)
            src, = _get_quantizers(srcmod, layer, head)
            if src == None or src.encoding == None:
                errors.append(f"Error in unify_kv_encodings: src==None: {_name(srcmod, layer, head)}, is_input:{srcmod.is_input}, index:{srcmod.index} ")
                continue
            for dstmod in dstmods:
                # for dst in _get_quantizers(dstmod, layer, head):
                stride = 1
                if itr_cnt%2==1:
                    stride = 6
                else:
                    stride = 1
                print(f"for dst in _get_quantizers(dstmod, layer, head//{stride}): ##Hardcoded")
                for dst in _get_quantizers(dstmod, layer, head//stride): ##Hardcoded
                    # print(" src.encoding.delta ", src.encoding.delta)
                    if dst == None or dst.encoding == None:
#                         errors.append(f"Error in unify_kv_encodings: dst==None: {_name(dstmod, layer, head)}, is_input:{dstmod.is_input}, index:{dstmod.index} ")
#                         continue
                        dst.enabled = True
                        dst.encoding = src.encoding
                        dst.use_symmetric_encodings = src.use_symmetric_encodings
                    else:
                        _copy(dst, src, dstmod.keep_bitwidth)

    assert not errors, '\n'.join(errors)


def prepare_combined_attention_mask(attention_mask, input_shape, device, past_key_values_length=0, dtype=torch.float32):
    from gauss2_torch.threeb.modeling_llama import LlamaModel
    dummy_enbedding = torch.tensor((1.0,)).to(dtype).to(device)
    new_mask = LlamaModel._prepare_decoder_attention_mask(attention_mask, input_shape, dummy_enbedding, past_key_values_length)
    return new_mask

def _is_separate_kv_head(model_cfg):
    separate_kv_head = False
    if hasattr(model_cfg, 'separate_kv_head'):
        separate_kv_head = model_cfg.separate_kv_head
    elif type(model_cfg).__name__ == 'LlamaConfig':
        separate_kv_head = True
    return separate_kv_head

def get_dummy_kvcache(batch_size, past_size, model_cfg, device, max_tokens=None):
    def _cache(shape):
        return torch.zeros(shape).to(device=device)

    separate_kv_head = _is_separate_kv_head(model_cfg)
    num_layers, num_heads, max_tokens_from_model, embed_dim = extract_info_from_model_cfg(model_cfg)
    max_tokens = max_tokens if max_tokens is not None else max_tokens_from_model
    head_dim = 1 if separate_kv_head else num_heads
    value = (batch_size, head_dim, past_size, embed_dim//num_heads)
    key = (batch_size, head_dim, embed_dim//num_heads, past_size) if model_cfg.transposed_key_cache \
                            else (batch_size, head_dim, past_size, embed_dim//num_heads)

    if separate_kv_head:
        past_key_values = tuple((
                tuple(_cache(key) for _ in range(num_heads)),
                tuple(_cache(value) for _ in range(num_heads)),
            ) for _ in range(num_layers))
    else:
        past_key_values = tuple((_cache(key), _cache(value)) for _ in range(num_layers))
    return past_key_values 


def zero_lora_weight_appender(lora_supported_layers,num_decs, num_heads,hidden_state_size,rank,head_dim,intermediate_size):

    
    lora_weight_lists = {}

    for gen in lora_supported_layers:
        lora_weight_lists[f'{gen}_lora_A_weights'] = []
        lora_weight_lists[f'{gen}_lora_B_weights'] = []

    ## Generic-QKV-LoRa-section-starts
    for gen in lora_supported_layers:

        if gen not in ['q','k','v']:
            continue

        gen_lora_A_weights = ()
        gen_lora_B_weights = ()
        tail_hidden_state_size = hidden_state_size
        if gen in ['k','v']:
            tail_hidden_state_size = num_heads*head_dim

        for i in range(num_decs):
            gen_lora_A_weights+=(torch.randn(hidden_state_size,rank).cuda(),)
            gen_lora_B_weights+=(torch.randn(rank,tail_hidden_state_size).cuda(),)

        # for i in range(num_decs):
        #     temp_gen = ()
        #     for j in range(num_heads):
        #         temp_gen+=(torch.randn(rank,head_dim).cuda(),)

        #     gen_lora_B_weights+=(temp_gen,)

        lora_weight_lists[f"{gen}_lora_A_weights"].append(gen_lora_A_weights)
        lora_weight_lists[f"{gen}_lora_B_weights"].append(gen_lora_B_weights)

    ## Generi-QKVc-LoRa-section-ends
    
    ## Generic-LoRa-section-starts

    for gen in lora_supported_layers:

        if gen in ['q','k','v']:
            continue
        gen_lora_A_weights = ()
        gen_lora_B_weights = ()

        if gen in ['gate','up']:
            dim1 = hidden_state_size
            dim2 = intermediate_size
        elif gen == 'down':
            dim1 = intermediate_size
            dim2 = hidden_state_size
        else:
            dim1 = hidden_state_size
            dim2 = hidden_state_size

        for i in range(num_decs):
            gen_lora_A_weights+=(torch.randn(dim1,rank).cuda(),)
            gen_lora_B_weights+=(torch.randn(rank, dim2).cuda(),)
        
        lora_weight_lists[f"{gen}_lora_A_weights"].append(gen_lora_A_weights)
        lora_weight_lists[f"{gen}_lora_B_weights"].append(gen_lora_B_weights)
        
        ## Generic-LoRa-section-ends
        # print(lora_weight_lists)
    return lora_weight_lists





def _get_dummy_input(input_ids, attention_mask, device, past_size=0, batch_size=1, export_mode=None, model_cfg=None, max_input_length=None,num_of_lora_layers=30):
    num_layers, num_head, max_input_length_from_model, embed_dim = extract_info_from_model_cfg(model_cfg)
    num_head = model_cfg.num_key_value_heads
    if max_input_length is None:
        max_input_length = max_input_length_from_model

    input_length = len(input_ids[0])
    if export_mode != 'kvcache' and model_cfg.enable_preamble:
        past_size += model_cfg.preamble_size
    pad_size = max_input_length - past_size - input_length 

    padded_input_ids = torch.cat((torch.tensor([[0]*pad_size for _ in range(batch_size)], device=device).to(torch.int64), input_ids.to(device)), dim=1).to(device)
    padded_attention_mask = torch.cat((torch.tensor([[0] * (pad_size+past_size) for _ in range(batch_size)], device=device), attention_mask.to(device)), dim=1).to(device)

    position_ids = torch.cumsum(padded_attention_mask[:,past_size:], dim=1).to(torch.int64).to(device).clip(0, max_input_length-1).to(device)

    assert padded_input_ids.shape[-1] == position_ids.shape[-1], (padded_input_ids.shape, position_ids.shape)
    assert padded_attention_mask.shape[-1] == max_input_length

    if model_cfg.use_position_embedding_input:
        position_ids = RopeEmbedding(device=device,head_dim=model_cfg.hidden_size//model_cfg.num_attention_heads).get_embedding(position_ids)

    if model_cfg.use_combined_mask_input:
        padded_attention_mask = prepare_combined_attention_mask(padded_attention_mask, padded_input_ids.shape, device, past_key_values_length=past_size)

    if model_cfg.use_input_embedding_input:
        padded_input_ids = torch.unsqueeze(padded_input_ids, -1)
        padded_input_ids = padded_input_ids.expand(-1, -1, embed_dim).to(torch.float32)

    inputs = {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'position_ids': position_ids,
    }

    num_decs = num_layers
    num_heads = num_head

    global lora_supported_layers_list_global
    global lora_rank

    hidden_state_size = getattr(model_cfg, 'hidden_size')
    intermediate_size = getattr(model_cfg, 'intermediate_size')
    num_attention_heads = getattr(model_cfg, 'num_attention_heads')
    rank = lora_rank
    head_dim = hidden_state_size//num_attention_heads
    
    lora_supported_layers = lora_supported_layers_list_global

    lora_weight_list = zero_lora_weight_appender(lora_supported_layers,num_of_lora_layers, num_heads, hidden_state_size,rank,head_dim,intermediate_size)

    for gen in lora_supported_layers: 
        inputs[f'{gen}_lora_A_weights'] = lora_weight_list[f'{gen}_lora_A_weights'][0]
        inputs[f'{gen}_lora_B_weights'] = lora_weight_list[f'{gen}_lora_B_weights'][0]


    if export_mode == 'kvcache':
        inputs['past_key_values'] = get_dummy_kvcache(batch_size, past_size, model_cfg, device)
    elif model_cfg.enable_preamble:
        inputs['past_key_values'] = get_dummy_kvcache(batch_size, model_cfg.preamble_size, model_cfg, device)

    # print("dummy_inputs_returned,lora_supported_layers: ",lora_supported_layers,inputs)
    return inputs

def get_dummy_input_for_causal_llm(input_string:str, tokenizer, device, export_mode, model_cfg, max_tokens=None):
    if max_tokens is None:
        _, _, max_tokens, *_ = extract_info_from_model_cfg(model_cfg)
    def encode(input, add_special_tokens=False, return_tensors='pt', max_length=max_tokens, **kwargs):
        """Redirect to the encode method on the tokenizer"""
        encoded_tensor = tokenizer(input, add_special_tokens=add_special_tokens, return_tensors=return_tensors,
                                   max_length=max_length, truncation=True, **kwargs)
        return encoded_tensor

    max_input_length = max_tokens
    # FIXME: compare generation of dummy input with prepare_inputs_for_prepared_model
    if export_mode == 'kvcache':
        '''
        FIXME: Do we need this for preparer-pro?
        past_size==1 is to start the first inference with the dummy 'past_key_values'
        And it was needed because we have to provide 'past_key_values' input always for traced models
        '''
        past_size = 1
        encoded = encode(input=input_string, max_length=max_input_length-past_size)
        return _get_dummy_input(encoded.input_ids, encoded.attention_mask, device, past_size=past_size, export_mode=export_mode, model_cfg=model_cfg, max_input_length=max_tokens)
    else:
        encoded = encode(input=input_string, max_length=max_input_length-(model_cfg.preamble_size if model_cfg.enable_preamble else 0))
        return _get_dummy_input(encoded.input_ids, encoded.attention_mask, device, export_mode=export_mode, model_cfg=model_cfg, max_input_length=max_tokens)

    return dummy_input

def get_dummy_input_for_causal_llm_export(export_mode, model_cfg, device='cpu', max_tokens=None,num_of_lora_layers=30):
    if max_tokens is None:
        _, _, max_tokens, *_ = extract_info_from_model_cfg(model_cfg)
    if export_mode == 'kvcache':
        # NOTE: support SpeculativeDecodinu
        num_input_tokens = 1 if model_cfg.num_logits_to_return <= 1 else model_cfg.num_logits_to_return-1
        past_size = max_tokens - num_input_tokens
        input_ids = torch.ones((1,num_input_tokens),dtype=torch.int64).to(device=device)
        attention_mask = torch.ones(1, num_input_tokens).to(device=device)
        return _get_dummy_input(input_ids, attention_mask, device, past_size=past_size, export_mode=export_mode, model_cfg=model_cfg, max_input_length=max_tokens,num_of_lora_layers=num_of_lora_layers)
    else:
        input_ids = torch.ones((1,max_tokens-(model_cfg.preamble_size if model_cfg.enable_preamble else 0)),dtype=torch.int64).to(device=device)
        attention_mask = torch.ones(1, max_tokens-(model_cfg.preamble_size if model_cfg.enable_preamble else 0)).to(device=device)
        return _get_dummy_input(input_ids, attention_mask, device, export_mode=export_mode, model_cfg=model_cfg, max_input_length=max_tokens,num_of_lora_layers=num_of_lora_layers)

    return dummy_input

def get_input_output_names(export_mode, model_cfg,num_lora_layers):
    if type(model_cfg).__name__ == 'LlamaConfig':
        return _get_input_output_names_for_llama(export_mode, model_cfg,num_lora_layers)
    else:
        return _get_input_output_names_for_gpt(export_mode, model_cfg)

def _get_input_output_names_for_gpt(export_mode, model_cfg):
    num_layers, _, _, _ = extract_info_from_model_cfg(model_cfg)
    def _get_past_key_values_names(sfx, n_layers):
        all = []
        for i in range(n_layers):
            all.append(f'past_key_{i}_{sfx}')
            all.append(f'past_value_{i}_{sfx}')
        return all

    if export_mode == 'kvcache':
        input_names = ['input_ids'] + _get_past_key_values_names('in', num_layers) + ['attention_mask', 'position_ids']
        output_names = ['logits'] + _get_past_key_values_names('out', num_layers)
    elif export_mode == 'bertcache':
        input_names = ['input_ids', 'attention_mask', 'position_ids']
        output_names = ['logits'] + _get_past_key_values_names('out', num_layers)
    else:
        assert False, f'Unexpected export mode:{export_mode}'
    return input_names, output_names

def _get_input_output_names_for_llama(export_mode, model_cfg,num_lora_layers):
    num_layers, num_head, _, _ = extract_info_from_model_cfg(model_cfg)
    num_head = model_cfg.num_key_value_heads
    def _get_past_key_values_names(sfx, n_layers, n_heads):
        all = []
        for i in range(n_layers):
            all.extend([f'past_key_{i}_h{h}_{sfx}' for h in range(n_heads)])
            all.extend([f'past_value_{i}_h{h}_{sfx}' for h in range(n_heads)])
        return all

    def _get_lora_weight_names(n_layers, n_heads):
        all = []

        global lora_supported_layers_list_global
        lora_supported_layers = lora_supported_layers_list_global

        for gen in lora_supported_layers:

            if gen in ['q','k','v']:
                all.extend([f'{gen}_lora_A_weights_d_{i}' for i in range(n_layers)])
                all.extend([f'{gen}_lora_B_weights_d_{i}' for i in range(n_layers)])
                # for i in range(n_layers):
                #     all.extend([f'{gen}_lora_B_weights_d_{i}_h_{h}' for h in range(n_heads)])
            else:
                all.extend([f'{gen}_lora_A_weights_d_{i}' for i in range(n_layers)])
                all.extend([f'{gen}_lora_B_weights_d_{i}' for i in range(n_layers)])
        print("LoRa weight names: ",all)
        return all


    def _get_position_emb_names():
        if model_cfg.use_position_embedding_input:
            return ['position_ids_cos', 'position_ids_sin']
        return ['position_ids']

    if export_mode == 'kvcache': #export_mode == 'kvcache_v1' or  export_mode == 'kvcache_v2' or model_cfg.enable_preamble:
        
        if model_cfg.use_input_embedding_input:
            input_names = ['inputs_embeds', 'attention_mask']
        else:
            input_names = ['input_ids', 'attention_mask']
        input_names += _get_position_emb_names()
        input_names += _get_lora_weight_names( num_lora_layers, num_head)
        input_names += _get_past_key_values_names('in', num_layers, num_head)
        output_names = ['logits']
        if model_cfg.return_top_k > 0:
            output_names += ['indices']
        output_names += _get_past_key_values_names('out', num_layers, num_head)
    elif export_mode == 'bertcache':
        
        if model_cfg.use_input_embedding_input:
            input_names = ['inputs_embeds', 'attention_mask']
        else:
            input_names = ['input_ids', 'attention_mask']
        input_names += _get_position_emb_names()
        input_names += _get_lora_weight_names( num_lora_layers, num_head)
        output_names = ['logits']
        if model_cfg.return_top_k > 0:
            output_names += ['indices']
        output_names += _get_past_key_values_names('out', num_layers, num_head)
    else:
        assert False, f'Unexpected export mode:{export_mode}'
    return input_names, output_names

def to_device(t, device):
    if isinstance(t, torch.Tensor):
        return t.detach().clone().to(device)
    if isinstance(t, tuple):
        return tuple([to_device(i, device) for i in t])
    if isinstance(t, list):
        return [to_device(i, device) for i in t]
    if isinstance(t, dict):
        return {k:to_device(v, device) for k,v in t.items()}
    return t

def to_cpu(t):
    return to_device(t, torch.device('cpu'))


class ForwardHook:
    def __init__(self):
        from collections import defaultdict
        self.data = defaultdict(dict)

    def get_activation_stats(self, name, types):
        def hook(model, input, output):
            for t in types:
                if t == "input":
                    x = input
                elif t == "output":
                    x = output
                else:
                    raise Exception("Please only assign input/output as value in your config file")

                if type(x) is tuple:
                    x = x[0]
                stats = (x.min().item(), x.max().item(), x.mean().item())
                self.data[name][t] = stats

        return hook

    def get_activation(self, name, types):
        def hook(model, input, output):
            for t in types:
                if t == "input":
                    # FIXME: Always 1st input only
                    self.data[name][t] = to_cpu(input[0] if isinstance(input, (tuple,list)) else input)
                elif t == "output":
                    self.data[name][t] = to_cpu(output[0] if isinstance(output, (tuple,list)) else output)
                else:
                    raise Exception("Please only assign input/output as value in your config file")
        return hook


def get_sqnr(fp_out, qt_out, eps=1e-10):
    quant_error = fp_out - qt_out
    exp_noise = (quant_error ** 2).mean() + eps
    exp_signal = (fp_out ** 2).mean()
    sqnr = (exp_signal / exp_noise)
    sqnr_db = 10 * np.log10(sqnr)
    return sqnr_db


def compute_sqnrs(fp_data, qt_data, config_data):
    logs = []
    for (k_fp, v_fp), (k_qt, v_qt) in zip(fp_data.items(), qt_data.items()):
        if k_fp != k_qt:
            print(f"Key mismatch: {k_fp} and {k_qt}")
            break
        if k_fp in config_data.keys():
            # intermediate tensors
            for tensor_type in config_data[k_fp]:
                if isinstance(v_fp[tensor_type], list) and isinstance(v_qt[tensor_type], list):
                    assert len(v_fp[tensor_type]) == len(v_qt[tensor_type])
                    for i in range(len(v_fp[tensor_type])):
                        sqnr_db = get_sqnr(v_fp[tensor_type][i], v_qt[tensor_type][i])
                        log = "SQNR of {}_{}: {:.1f} dB".format(k_fp+'_'+tensor_type, i, sqnr_db)
                        print(log)
                        logs.append(log)
                else:
                    sqnr_db = get_sqnr(v_fp[tensor_type], v_qt[tensor_type])
                    log = "SQNR of {}: {:.1f} dB".format(k_fp+'_'+tensor_type, sqnr_db)
                    print(log)
                    logs.append(log)
        elif k_fp in ['output_key_values', 'past_key_values']:
            assert len(v_fp) == len(v_qt)
            for i in range(len(v_fp)):
                key_sqnr_db = get_sqnr(v_fp[i][0], v_qt[i][0])
                value_sqnr_db = get_sqnr(v_fp[i][1], v_qt[i][1])
                log = "SQNR of {}_{}: {:.1f} dB and {}_{}: {:.1f} dB".format(k_fp.split('_')[0]+"_keys", i, key_sqnr_db, k_fp.split('_')[0]+"_values", i, value_sqnr_db)
                print(log)
                logs.append(log)
        else:
            # input_ids, attention_mask, position_ids, logits
            sqnr_db = get_sqnr(v_fp.type(torch.float32), v_qt.type(torch.float32))
            log = "SQNR of {}: {:.1f} dB".format(k_fp, sqnr_db)
            print(log)
            logs.append(log)

    return logs


def evaluate_test_vectors(args):
    num_samples = args.num_test_vectors * args.per_device_eval_batch_size
    config_data = json.load(open(f"config/{args.model_name}_hooks.json", "rb"))

    for i in range(num_samples):
        fp_data = pickle.load(open(f"{args.export_dir}/test_vectors/fp_{i}.pkl", "rb"))[str(i)]
        qt_data = pickle.load(open(f"{args.export_dir}/test_vectors/qt_{i}.pkl", "rb"))[str(i)]

        logs = compute_sqnrs(fp_data, qt_data, config_data)

        with open(f"{args.export_dir}/test_vectors/sqnrs_{i}.txt", "w") as f:
            for log in logs:
                f.write(log+'\n')

class RopeEmbedding:
    def __init__(self, device, head_dim=80, max_length=2048, theta=500000.0):
        self.max_length = max_length
        self.cos, self.sin = self.precompute_freqs_cis(head_dim, max_length * 2, theta=theta, device=device)

    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 500000.0, device=None):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        freqs_cis = freqs_cis[0:self.max_length]
        freqs_real = torch.view_as_real(freqs_cis)
        freqs_real = freqs_real.unsqueeze(0).unsqueeze(0)

        freqs_cos = freqs_real[:,:,:,:,0] # extract even elements
        freqs_sin = freqs_real[:,:,:,:,1] # extract odd elements
        return freqs_cos.to(device), freqs_sin.to(device)

    def get_embedding(self, position_ids):
        '''
        position_ids: [batch_size, sequence_length]
        return [batch_size, 1, sequence_length, head_sim//2][2]
        '''
        cos = self.cos[0,0,:,:]  # [seq_len, dim]
        sin = self.sin[0,0,:,:]  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
        return cos, sin

def dump_config_json(filename, config):
    with open(filename, 'wt') as f:
        json.dump(vars(config), f, indent=2, default=lambda x:f'{x}')

def evaluate(prepared_model, tokenizer, model_quantizer, dataset_builder, args):
    metric = "perplexity"
    print(f"Evaluating {metric}")
    # 4 types of prepared_model are evaluated (args.prepare_model / (args.use_ptq or args.use_qat))
    #   1. original FP model : (False / False)
    #   2. prepared FP model : (True / False)
    #   3. quantsim of original model : (False / True)
    #   4. quantsim of prepared model : (True / True)
    res, _ = model_quantizer.evaluate(
        model=prepared_model,
        iterations=1e10,
        loader=dataset_builder.test_dataloader,
        tokenizer=tokenizer,
        metric=metric,
        do_eval=True,
        enable_preamble=args.eval_with_preamble,
        full_mode=False,
    )
    eval_loss = res["loss"]
    perplexity = res[metric]

    print(f"{metric}: {perplexity} eval_loss: {eval_loss}")
    if args.output_dir is not None:
        result_dir = os.path.join(
            args.output_dir, datetime.now().strftime("%Y%m%d%H%M%S%f")
        )
        os.makedirs(result_dir, exist_ok=True)
        dump_config_json(os.path.join(result_dir, "cfg_args.json"), args)
        with open(os.path.join(result_dir, f"all_results.{args.do_prepare}.{args.do_quantize}.{args.mask_neg}.json"), "w") as f:
            json.dump({"perplexity": perplexity, "loss": eval_loss}, f, ensure_ascii=False, indent=2)