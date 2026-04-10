# qwen3-finetune-main
You can get the data set used in the paper at the following links:https://pan.quark.cn/s/a511e76659e5

This project is mainly used for **Qwen3 model fine-tuning and inference**. The core scripts currently visible in this repository are:

- `qwen3_finetune.py`: a LoRA fine-tuning script based on **ModelScope Swift**
- `finetune/finetune_lora_single_gpu.sh`: a single-GPU LoRA fine-tuning launch script
- `qwen3_inference.py`: loads the base model and LoRA adapters for inference/evaluation
- `demo.py`: a small utility script that converts raw data into the training format

Since the repository does **not include `requirements.txt`, `pyproject.toml`, or `environment.yml`**, and several scripts contain **machine-specific absolute paths**, you need to prepare the Python environment, model path, and dataset path before running the project.

---

## 1. Environment Requirements

Recommended environment:

- **OS**: Windows 10/11 or Linux
- **Python**: `3.10` or `3.11` recommended
- **CUDA**: `11.8` recommended, or a version compatible with your local PyTorch build
- **GPU**: NVIDIA GPU recommended; the current scripts can run on a single GPU
- **VRAM**:
  - For `Qwen3-0.6B` + LoRA fine-tuning, **12GB+** is recommended
  - If `max_length=131072` remains unchanged, memory usage will be very high, so reducing it based on your hardware is strongly recommended

---

## 2. Required Dependencies

Based on the current scripts, you will need at least the following Python packages.

### Core dependencies

```bash
pip install torch torchvision torchaudio
pip install modelscope ms-swift transformers datasets accelerate peft
```

### Additional dependencies used by the project

```bash
pip install matplotlib nltk pandas numpy pymupdf requests openai
```

If you want to inspect training logs with TensorBoard, it is also recommended to install:

```bash
pip install tensorboard
```

### Optional NLTK resource

`qwen3_inference.py` uses `nltk.translate.bleu_score`, which usually does not require extra corpora. If NLTK raises a resource-related error on your machine, run:

```bash
python -c "import nltk; nltk.download('punkt')"
```

---

## 3. Recommended: Create a Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Windows Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies after activating the environment.

---

## 4. Paths You Must Modify Before Running

The current scripts contain many hard-coded absolute paths, so running them directly will likely fail. At minimum, you need to update the following items.

### 4.1 Fine-tuning script `qwen3_finetune.py`

Key configuration locations:

- `qwen3_finetune.py:3`: `CUDA_VISIBLE_DEVICES`
- `qwen3_finetune.py:15`: base model path `model_id_or_path`
- `qwen3_finetune.py:20`: training dataset path `dataset`
- `qwen3_finetune.py:16`: output directory `output_dir`

The original script looks like this:

```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_id_or_path = '<your-local-base-model-path>'
dataset = ['<your-local-training-dataset-path>']
output_dir = 'output'
```

You need to replace these with paths on your own machine. In general, you can simply use your local base model directory and your local training dataset file path.

---

### 4.2 Shell launch script `finetune/finetune_lora_single_gpu.sh`

Key configuration locations:

- `finetune/finetune_lora_single_gpu.sh:4`: model path `MODEL`
- `finetune/finetune_lora_single_gpu.sh:7`: dataset path `DATA`

The original script looks like this:

```bash
MODEL="<your-local-base-model-path>"
DATA="<your-local-training-dataset-path>"
```

Replace them with your own local paths.

**Note:**
Although this is a `.sh` file, it currently uses **Windows-style paths**. If you run it on Linux, you must change them to Linux paths. If you run it in Windows Git Bash, it is also recommended to normalize them to `/c/...` or manage paths directly inside Python scripts to avoid escaping issues.

---

### 4.3 Inference script `qwen3_inference.py`

Key configuration locations:

- `qwen3_inference.py:2`: `CUDA_VISIBLE_DEVICES`
- `qwen3_inference.py:16`: LoRA adapter paths `adapter_path`
- `qwen3_inference.py:17`: base model path `model_id_or_path`
- `qwen3_inference.py:245`: evaluation dataset directory `read_files(...)`

In the original script:

```python
adapter_path = []
model_id_or_path = '<your-local-base-model-path>'
method_bodies, method_names = read_files('<your-local-evaluation-dataset-directory>')
```

At minimum, you need to:

1. Change `model_id_or_path` to your local base model directory
2. Change `adapter_path` to your list of LoRA checkpoint directories
3. Change the evaluation dataset directory to your own dataset path

---

## 5. Data Format

### 5.1 Fine-tuning data

Based on `finetune/finetune_lora_single_gpu.sh` and `demo.py`, the training data should be a **JSON file** containing a **list of conversations** or another Swift-compatible format.

`demo.py` converts raw samples into a structure like this:

```json
[
  {
    "messages": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }
]
```

So your training data is recommended to follow a similar JSON structure.

### 5.2 Inference / evaluation data

`qwen3_inference.py` depends on a directory-based dataset. The script recursively reads files in a dataset folder and looks for:

- `*methodBodies.txt`
- `*methodNames.txt`

If your dataset does not follow this structure, you need to either adjust the script or prepare the dataset accordingly.

---

## 6. How to Start the Project

There are three main ways to run this project.

### Option 1: Run the Python fine-tuning script directly

After updating the model path and dataset path in `qwen3_finetune.py`, run:

```bash
python qwen3_finetune.py
```

This script will:

1. Load the base model
2. Build the LoRA configuration
3. Read training/validation data
4. Start training
5. Save outputs to `output/`

Training artifacts are saved by default under:

```text
output/
```

---

### Option 2: Run the Swift single-GPU fine-tuning script

First, update:

- `MODEL` in `finetune/finetune_lora_single_gpu.sh`
- `DATA` in `finetune/finetune_lora_single_gpu.sh`

Then run:

```bash
bash finetune/finetune_lora_single_gpu.sh
```

This script essentially runs:

```bash
swift sft ...
```

So `ms-swift` must already be installed and the `swift` command must be available.

If the command is not available, check with:

```bash
swift --help
```

---

### Option 3: Run the inference / evaluation script

After updating the following in `qwen3_inference.py`:

- base model path
- LoRA checkpoint path(s)
- benchmark dataset directory

run:

```bash
python qwen3_inference.py
```

The script will read the evaluation data in batch, perform method-name prediction, and print evaluation metrics such as:

- Precision
- Recall
- F-score
- Exact Match

---

## 7. Common Issues

### 7.1 File not found when running the scripts

This is usually caused by hard-coded paths from the original author’s local machine.

Check these locations first:

- `qwen3_finetune.py:15`
- `qwen3_finetune.py:20`
- `qwen3_inference.py:17`
- `qwen3_inference.py:245`
- `finetune/finetune_lora_single_gpu.sh:4`
- `finetune/finetune_lora_single_gpu.sh:7`

---

### 7.2 `swift: command not found`

This means `ms-swift` was not installed correctly, or your virtual environment is not activated.

Try:

```bash
pip install ms-swift
swift --help
```

If it still does not work, use:

```bash
python qwen3_finetune.py
```

instead.

---

### 7.3 CUDA / PyTorch version mismatch

If you encounter issues such as GPU unavailable or CUDA initialization failure, first check:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Make sure your installed PyTorch build matches your local CUDA environment.

---

### 7.4 `TkAgg` backend errors

`qwen3_finetune.py:103` uses:

```python
matplotlib.use('TkAgg')
```

If your environment does not have a GUI, this may fail. Possible fixes:

- install Tk-related dependencies
- or switch the backend to a non-GUI backend such as `Agg`

---

### 7.5 Out-of-memory caused by very large context length

The current scripts set very large sequence lengths:

- `qwen3_finetune.py:21` sets `max_length = 131072`
- `finetune/finetune_lora_single_gpu.sh:110` uses `--max_length 131072`
- `qwen3_inference.py:18` sets `max_length = 131072`

This is extremely large. If you run into VRAM issues, try reducing it first to something like:

- `2048`
- `4096`
- `8192`

Then adjust further based on your hardware.

---

## 8. Recommended Startup Order

A recommended setup and startup order is:

1. Create and activate a virtual environment
2. Install dependencies
3. Download or prepare the base model
4. Prepare training data / evaluation data
5. Update the hard-coded paths in the scripts
6. Run fine-tuning first:

```bash
python qwen3_finetune.py
```

or:

```bash
bash finetune/finetune_lora_single_gpu.sh
```

7. After training finishes, fill in the LoRA checkpoint path and then run:

```bash
python qwen3_inference.py
```

---

## 9. Confirmed Key Files in This Repository

- `qwen3_finetune.py:15`: base model path
- `qwen3_finetune.py:20`: training dataset path
- `qwen3_finetune.py:101`: training entry point
- `qwen3_inference.py:16`: LoRA adapter path configuration
- `qwen3_inference.py:17`: base model path
- `qwen3_inference.py:245`: evaluation dataset directory
- `finetune/finetune_lora_single_gpu.sh:92`: Swift SFT launch command
- `demo.py:65`: data conversion script entry point

---

## 10. Minimal Working Example

Assume you have already:

- installed the dependencies
- downloaded the base model
- prepared your training dataset file

Then the minimal startup process is:

### Step 1: Modify `qwen3_finetune.py`

```python
model_id_or_path = '<your-local-base-model-path>'
dataset = ['<your-local-training-dataset-path>']
output_dir = 'output'
```

### Step 2: Start training

```bash
python qwen3_finetune.py
```

### Step 3: After training, modify `qwen3_inference.py`

```python
adapter_path = ['<your-local-lora-checkpoint-path>']
model_id_or_path = '<your-local-base-model-path>'
```

### Step 4: Start inference / evaluation

```bash
python qwen3_inference.py
```

---

If you want, I can continue and help you with one of the following next:

1. a **Windows-specific startup guide**
2. a **Linux server startup guide**
3. a generated `requirements.txt`
