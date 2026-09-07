import torch

from modelscope import AutoTokenizer, AutoModelForCausalLM

end_id = 151668
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
messages = [
    {
        "role": "user",
        "content": "小明的爸爸叫大明，大明的爸爸叫老明，请问小明的爷爷叫什么？",
    }
]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    enable_thinking=True,
    add_generation_prompt=True,
)
inputs = tokenizer(
    [prompt],
    return_tensors="pt",
).to(model.device)
# inputs = tokenizer.apply_chat_template(
#     messages,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt",
#     enable_thinking=True,
#     add_generation_prompt=True,
# ).to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=1024)
output_ids = generated_ids[0][len(inputs.input_ids[0]) :].tolist()
if end_id in output_ids:
    index = len(output_ids) - output_ids[::-1].index(end_id)
else:
    index = 0
thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
print(f"prompt:\n{prompt}")
print(f"thinking:\n{thinking}")
print(f"content:\n{content}")
