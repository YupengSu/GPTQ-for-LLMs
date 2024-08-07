# GPTQ-for-LLMs
GPTQ Algorithm Repository Reconstruction for LLM Quantization.

## Installation
```
conda create -n quant_llm python=3.9
conda activate quant_llm
```

```
pip3 install torch torchvision torchaudio
pip3 install -r requirements.txt
```

## Usage
```
python main.py \
    --model /path/to/your/model \
    --dataset c4 \
    --wbits 4 \
    --act-order \
    --save_model /path/to/save/hf-model \
    --save_checkpoint /path/to/save/model-w4-gptq.pt \
    --save_ppl /path/to/save/ppl \
    --save_zeroshot /path/to/save/zeroshot
```