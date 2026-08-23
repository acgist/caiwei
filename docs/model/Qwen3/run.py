import time
import torch

from modelscope import AutoTokenizer, AutoModelForCausalLM


def get_memory():
    reserved = 0
    allocated = 0
    gpu_count = torch.cuda.device_count()
    for gpu_id in range(gpu_count):
        reserved += torch.cuda.memory_reserved(gpu_id)
        allocated += torch.cuda.memory_allocated(gpu_id)
    print(f"cuda memory reserved: {(reserved / (1024**3)):.2f} GB")
    print(f"cuda memory allocated: {(allocated / (1024**3)):.2f} GB")


end_id = 151668
model_name = "Qwen/Qwen3-4B"
torch_dtype = "auto"  # torch.float16
tt = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch_dtype,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"dtype: {model.dtype}")
print(f"load model: {(time.time() - tt):.2f} s")
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
tt = time.time()
generated_ids = model.generate(**inputs, max_new_tokens=1024)
output_ids = generated_ids[0][len(inputs.input_ids[0]) :].tolist()
if end_id in output_ids:
    index = len(output_ids) - output_ids[::-1].index(end_id)
else:
    index = 0
thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
print(f"generate: {(time.time() - tt):.2f} s")
get_memory()
print(f"prompt:\n{prompt}")
print(f"thinking:\n{thinking}")
print(f"content:\n{content}")
input("按回车键退出...")
