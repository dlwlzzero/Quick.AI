import sys


sys.path.append("/group-volume/loki.ravuru/NPUForge-main/aimet")
MODEL_PATH = "/group-volume/models/gemma-4-E2B-it-textonly-untied-foldedple"
IS_FOLDED = True
model_type = "gemma4_e2b"
QSIM_BASIC_CONFIG = "/group-volume/loki.ravuru/NPUForge-main/aimet/configs/qsim_config.json"
EXCEPTION_FILE = "/group-volume/loki.ravuru/NPUForge-main/aimet/configs/exceptions/default_simple_a16_exceptions.json"
MODE = "generation"

QUANTIZATION_OVERRIDES = "/group-volume/loki.ravuru/NPUForge-main/aimet/workdir/gemma-4-E2B-it-textonly-untied-foldedple_gemma4_e2b_foldedple_8k/onnx/gemma-4-E2B-it-textonly-untied-foldedple_generation_torch.json"

import qmodels
import helpers
from helpers.exceptions import ExceptionConfigurator
from helpers.aimet_analyzer_utils import weight_quant_context, activation_quant_context, no_quant_context, disable_uninitialized_quant_modules
from aimet_torch import QuantizationSimModel
from aimet_torch.common.defs import QuantScheme
from aimet_torch.quant_analyzer import QuantAnalyzer
import torch
from types import SimpleNamespace

from transformers import AutoProcessor
from gemma4_qc.gemma4_text import preprocess_input, postprocess_outputs



@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.device_count() > 0 else "cpu"
    instance, config, tokenizer = qmodels.load_model(model_type, MODEL_PATH, device=device)
    
    mode_inputs = instance.get_dummy_data(MODE, 1, 8192, config, device)
    
    instance.prepare_forward(MODE)
    args = SimpleNamespace(activation_bitwidth=16, exceptions_file=EXCEPTION_FILE)
    exception_configurator = ExceptionConfigurator(args)
    exception_configurator.strict = False
    
    sim = QuantizationSimModel(
        model=instance,
        quant_scheme=QuantScheme.post_training_tf,
        dummy_input=helpers.to_device(mode_inputs, device),
        default_output_bw=16,
        default_param_bw=4,
        config_file=QSIM_BASIC_CONFIG,
        in_place=True,
    )

    exception_configurator.apply_pre_calibration_exceptions(sim)
    instance.pre_calibration_callback(sim, config)
    
    sim.load_encodings(QUANTIZATION_OVERRIDES, strict=False, partial=True, allow_overwrite=False)
    
    instance.post_calibration_callback(sim, config)
    exception_configurator.apply_post_calibration_exceptions(sim)
    
    # sim is now proper sim
    disable_uninitialized_quant_modules(sim)
    
    # input_text = "What is the capital of South Korea?"
    input_text = "무지개를 주제로 한 짧은 글을 써 줘."
    # input_text = "What is the rainbow?"
    # input_text = "What is 1/3 + 1/4?"
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    inputs = preprocess_input(input_text, processor, device=device)
    input_ids = inputs["input_ids"]
    input_ids = input_ids.to(device)
    sim.model.return_dict = True
    sim.model.config.return_dict = True
    sim.model.forward = sim.model.forward_conv

    all_quant_modules = QuantAnalyzer._get_quantized_modules(sim)

    # fp32_score, weightquant_score, actquant_score = QuantAnalyzer.check_model_sensitivity_to_quantization(sim)
    print("*****  No Quant Inference ***** \n\n")
    with no_quant_context(all_quant_modules):
        with torch.no_grad():
            generated_conv = sim.model.generate(input_ids, max_new_tokens=50, do_sample=False)
        # print(f"  new tokens (conv): {generated_conv[0, input_ids.shape[1]:].tolist()}")
        print(f"  generated text (conv) : {postprocess_outputs(inputs, generated_conv, processor)[0]}")

    print("*****  Only Weight Quant Inference ***** \n\n")
    with weight_quant_context(all_quant_modules):
        with torch.no_grad():
            generated_conv = sim.model.generate(input_ids, max_new_tokens=50, do_sample=False)
        # print(f"  new tokens (conv): {generated_conv[0, input_ids.shape[1]:].tolist()}")
        print(f"  generated text (conv) : {postprocess_outputs(inputs, generated_conv, processor)[0]}")


    print("*****  Only Activation Quant Inference ***** \n\n")
    with activation_quant_context(all_quant_modules):
        with torch.no_grad():
            generated_conv = sim.model.generate(input_ids, max_new_tokens=50, do_sample=False)
        # print(f"  new tokens (conv): {generated_conv[0, input_ids.shape[1]:].tolist()}")
        print(f"  generated text (conv) : {postprocess_outputs(inputs, generated_conv, processor)[0]}")

    print("*****  All Quant Inference ***** \n\n")
    with torch.no_grad():
        generated_conv = sim.model.generate(input_ids, max_new_tokens=50, do_sample=False)
    # print(f"  new tokens (conv): {generated_conv[0, input_ids.shape[1]:].tolist()}")
    print(f"  generated text (conv) : {postprocess_outputs(inputs, generated_conv, processor)[0]}")


main()
