"""Shared GGUF model catalog for the SLM benchmark notebooks."""
from pathlib import Path

MODEL_CATALOG = [
    {
        "name": "SmolLM2-135M-Instruct",
        "hf_repo": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "repo": "lmstudio-community/SmolLM2-135M-Instruct-GGUF",
        "file": "SmolLM2-135M-Instruct-Q4_K_M.gguf",
        "params_b": 0.135,
        "approx_disk_mb": 105,
        "note": "Baseline nhỏ nhất.",
    },
    {
        "name": "Qwen2.5-0.5B-Instruct",
        "hf_repo": "Qwen/Qwen2.5-0.5B-Instruct",
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "params_b": 0.5,
        "approx_disk_mb": 490,
        "note": "Baseline nhỏ.",
    },
    {
        "name": "Qwen3-0.6B",
        "hf_repo": "Qwen/Qwen3-0.6B",
        "repo": "bartowski/Qwen_Qwen3-0.6B-GGUF",
        "file": "Qwen_Qwen3-0.6B-Q4_K_M.gguf",
        "params_b": 0.6,
        "approx_disk_mb": 500,
        "note": "Qwen3 nhỏ nhất.",
    },
    {
        "name": "Llama-3.2-1B-Instruct",
        "hf_repo": "meta-llama/Llama-3.2-1B-Instruct",
        "repo": "lmstudio-community/Llama-3.2-1B-Instruct-GGUF",
        "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "params_b": 1.0,
        "approx_disk_mb": 770,
        "note": "Llama 3.2 nhỏ nhất dưới 3B; Hugging Face model gốc yêu cầu chấp nhận license Meta.",
    },
    {
        "name": "Gemma-3-1B-it",
        "hf_repo": "google/gemma-3-1b-it",
        "repo": "ggml-org/gemma-3-1b-it-GGUF",
        "file": "gemma-3-1b-it-Q4_K_M.gguf",
        "params_b": 1.0,
        "approx_disk_mb": 806,
        "note": "Gemma 3 nhỏ nhất dưới 3B; có thể cần chấp nhận điều khoản Gemma.",
    },
    {
        "name": "Arch-Agent-1.5B",
        "hf_repo": "katanemo/Arch-Agent-1.5B",
        "repo": "mradermacher/Arch-Agent-1.5B-GGUF",
        "file": "Arch-Agent-1.5B.Q4_K_M.gguf",
        "params_b": 1.5,
        "approx_disk_mb": 986,
        "note": "Fine-tune Qwen2.5-Coder; license riêng của Katanemo.",
    },
    {
        "name": "xLAM-2-1B-fc-r",
        "hf_repo": "Salesforce/xLAM-2-1b-fc-r",
        "repo": "Salesforce/xLAM-2-1b-fc-r-gguf",
        "file": "xLAM-2-1B-fc-r-Q4_K_M.gguf",
        "params_b": 1.0,
        "approx_disk_mb": 986,
        "note": "Model function-calling; license CC BY-NC 4.0 (phi thương mại).",
    },
    {
        "name": "Qwen3-1.7B",
        "hf_repo": "Qwen/Qwen3-1.7B",
        "repo": "bartowski/Qwen_Qwen3-1.7B-GGUF",
        "file": "Qwen_Qwen3-1.7B-Q4_K_M.gguf",
        "params_b": 1.7,
        "approx_disk_mb": 1350,
        "note": "Qwen3 cỡ 1.7B.",
    },
    {
        "name": "TinyLlama-1.1B-Chat",
        "hf_repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "params_b": 1.1,
        "approx_disk_mb": 670,
        "note": "Baseline nhẹ.",
    },
    {
        "name": "Arch-Agent-3B",
        "hf_repo": "katanemo/Arch-Agent-3B",
        "repo": "mradermacher/Arch-Agent-3B-GGUF",
        "file": "Arch-Agent-3B.Q4_K_M.gguf",
        "params_b": 3.0,
        "approx_disk_mb": 1930,
        "note": "Fine-tune Qwen2.5-Coder; license riêng của Katanemo.",
    },
    {
        "name": "Qwen3-4B",
        "hf_repo": "Qwen/Qwen3-4B",
        "repo": "Qwen/Qwen3-4B-GGUF",
        "file": "Qwen3-4B-Q4_K_M.gguf",
        "params_b": 4.0,
        "approx_disk_mb": 2382,
        "note": "Qwen3 4B theo yêu cầu; có thể vượt ngân sách RAM của UNO Q.",
    },
    {
        "name": "Phi-4-mini-instruct",
        "hf_repo": "microsoft/Phi-4-mini-instruct",
        "repo": "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
        "file": "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
        "params_b": 3.8,
        "approx_disk_mb": 2375,
        "note": "Bản nhỏ nhất của dòng Phi-4; vẫn lớn hơn 3B.",
    },
    {
        "name": "Gemma-4-E4B-it",
        "hf_repo": "google/gemma-4-E4B-it",
        "repo": "unsloth/gemma-4-E4B-it-GGUF",
        "file": "gemma-4-E4B-it-Q4_K_M.gguf",
        "params_b": 4.0,
        "approx_disk_mb": 4745,
        "note": "Bản Gemma 4 nhỏ nhất hiện có; yêu cầu llama.cpp có hỗ trợ Gemma 4 và điều khoản Gemma.",
    },
    {
        "name": "Phi-3.5-mini-instruct",
        "hf_repo": "microsoft/Phi-3.5-mini-instruct",
        "repo": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "file": "Phi-3.5-mini-instruct-Q4_K_M.gguf",
        "params_b": 3.8,
        "approx_disk_mb": 2200,
        "note": "Phi-3.5 Mini là model nhỏ nhất của dòng Phi-3.5; baseline đã có trong catalog cũ.",
    },
    {
        "name": "Llama-4-Scout-17B-16E-Instruct",
        "hf_repo": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "repo": "unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF",
        "files": [
            "Q4_K_M/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00001-of-00002.gguf",
            "Q4_K_M/Llama-4-Scout-17B-16E-Instruct-Q4_K_M-00002-of-00002.gguf",
        ],
        "params_b": 109.0,
        "active_params_b": 17.0,
        "approx_disk_mb": 62332,
        "manual_only": True,
        "note": "Scout là Llama 4 nhỏ nhất: 109B tổng (17B active), 2 shard khoảng 65 GB; license Meta, không phù hợp RAM 4 GB.",
    },
]


def download_gguf(model_name, cache_dir="model_cache"):
    """Download one catalog model's GGUF shard(s) into the shared cache."""
    from huggingface_hub import hf_hub_download

    entry = next((model for model in MODEL_CATALOG if model["name"] == model_name), None)
    if entry is None:
        raise KeyError(f"Unknown model in catalog: {model_name}")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    files = entry["files"] if "files" in entry else [entry["file"]]
    paths = [
        Path(hf_hub_download(repo_id=entry["repo"], filename=name, local_dir=str(cache)))
        for name in files
    ]
    return paths[0]


def local_gguf_models(cache_dir="model_cache", include_manual=False):
    """Return catalog models whose complete GGUF file set is already cached."""
    cache = Path(cache_dir)
    available = {}
    for entry in MODEL_CATALOG:
        if entry.get("manual_only") and not include_manual:
            continue
        files = entry["files"] if "files" in entry else [entry["file"]]
        paths = [cache / name for name in files]
        if all(path.is_file() for path in paths):
            available[entry["name"]] = str(paths[0])
    return available
