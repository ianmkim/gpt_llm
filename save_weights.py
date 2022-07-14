from transformers import AutoModelForCausalLM
import torch
import argparse

def save_weights(fp16:bool=False,bf16:bool=False):
    model_name = "EleutherAI/gpt-neox-20b"
    if fp16:
        model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.float16)
    elif bf16:
        model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    model.save_pretrained("./gptneox")

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", default=False,action="store_true")
    parser.add_argument("--bf16", default=False,action="store_true")
    args = parser.parse_args()
    if args.fp16:
        save_weights(fp16=True)
    elif args.bf16:
        save_weights(bf16=True)
    else:
        save_weights()    
