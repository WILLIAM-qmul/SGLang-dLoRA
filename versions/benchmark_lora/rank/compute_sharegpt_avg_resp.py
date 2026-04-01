# filepath: /workspace/sglang/benchmark/lora/rank/compute_sharegpt_avg_resp.py
from sglang.bench_serving import get_tokenizer, compute_sharegpt_avg_resp_len

MODEL = "/workspace/models/Llama-2-7b-hf"
DATASET = "/workspace/datasets/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json"

tokenizer = get_tokenizer(MODEL)
avg_resp_len = compute_sharegpt_avg_resp_len(
    dataset_path=DATASET,
    tokenizer=tokenizer,
    fixed_output_len=None,
    context_len=None,
    prompt_suffix="",
    apply_chat_template=False,
)

print("Final avg_resp_len =", avg_resp_len)