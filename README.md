# GPT NeoX 20B & OPT-30B

Forked from https://github.com/mallorbc/GPTNeoX20B_HuggingFace. Runs inference for GPT NeoX 20B and OPT-30B

## Requirements for GPT NeoX 20B

Ideally you have one or more GPUs that total 48GB of VRAM or more.

However, even if you don't you can still run the model it will just take much longer.  

For example, running with one 3090 rather than two would take around 10 minutes to generate 100 tokens vs 10-30 seconds if you ran it one two GPUs.

If you don't have enough VRAM you need to make sure you have enough RAM to make up for it.

If you want to download the weights the way I do, you need roughly at least 50GB of Vram for the float16 or bfloat16 weights. If you want to use bfloat16 you need to make sure your CPU and GPU support it.

## Requirements for OPT-30B

Requires one or more GPUs that total in 70GB of VRAM or more. Similar to GPT-NeoX, the model will still fit on the CPU but the OPTDecoderLayer is painfully slow using CPU inference.


## Dependencies
```pip3 install -r requirements.txt```


## Running
```
python3 main.py # for GPT NeoX 20B
python3 main_opt.py # for OPT-30B
```

There are two flags, each can be seen with ```-h```

Use the ```--fp16``` flag to load and save the weights in float16 mode.

Use the ```--bf16``` flag to load and save the weights in bfloat16 mode.

Use bfloat16 when you can as its better.

When running, the model will always be casted to bfloat16 unless your GPU/CPU can't handle it.  You may desire different behavior.
