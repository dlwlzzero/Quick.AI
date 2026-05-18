from transformers import Gemma4ForCausalLM, AutoTokenizer, AutoProcessor
import torch

device = "cuda" if torch.cuda.device_count() > 0 else "cpu"
MODEL_PATH = "/group-volume/models/gemma-4-E2B-it-textonly-untied/"

model = Gemma4ForCausalLM.from_pretrained(MODEL_PATH)
model.to(device)
tok = AutoTokenizer.from_pretrained(MODEL_PATH)

processor = AutoProcessor.from_pretrained(MODEL_PATH)
def preprocess_input(text, processor):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text},
    ]

    # Process input
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True, 
        enable_thinking=False
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    inputs.pop("mm_token_type_ids")
    return inputs

def postprocess_outputs(inputs, outputs, processor):
    input_len = inputs["input_ids"].shape[-1]
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    output = processor.parse_response(response)
    return output["content"], response

inputs = preprocess_input("What is the capital of South Korea?", processor)
print(inputs)
# Generate output
outputs = model.generate(**inputs, max_new_tokens=256)
print(outputs)
output = postprocess_outputs(inputs, outputs, processor)
print(output[0])
