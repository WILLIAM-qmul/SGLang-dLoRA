import json
import random
import numpy as np
from transformers import AutoTokenizer
from typing import Optional

ASSISTANT_SUFFIX = "Assistant:"

def remove_suffix(text: str, suffix: str) -> str:
    return text[: -len(suffix)] if text.endswith(suffix) else text

def is_file_valid_json(path):
    try:
        with open(path) as f:
            json.load(f)
        return True
    except Exception:
        return False

def filter_sharegpt_requests(
    dataset_path: str,
    tokenizer_name_or_path: str,
    fixed_output_len: Optional[int] = None,
    context_len: Optional[int] = None,
    prompt_suffix: Optional[str] = "",
    apply_chat_template=False,
):
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    if not is_file_valid_json(dataset_path):
        print(f"Dataset file not found or invalid: {dataset_path}")
        return

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)

    with open(dataset_path) as f:
        dataset = json.load(f)

    # 过滤掉对话轮数少于2的样本
    dataset = [
        data
        for data in dataset
        if len(data.get("conversations", data.get("conversation", []))) >= 2
    ]
    # 只保留前两轮
    dataset = [
        (
            data.get("conversations", data.get("conversation", []))[0]["value"],
            data.get("conversations", data.get("conversation", []))[1]["value"],
        )
        for data in dataset
    ]
    random.shuffle(dataset)

    filtered_count = 0
    for i in range(len(dataset)):
        prompt = dataset[i][0]
        if prompt_suffix:
            prompt = (
                remove_suffix(prompt, ASSISTANT_SUFFIX)
                + prompt_suffix
                + ASSISTANT_SUFFIX
            )

        if apply_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            if tokenizer.bos_token:
                prompt = prompt.replace(tokenizer.bos_token, "")

        prompt_token_ids = tokenizer.encode(prompt)
        completion = dataset[i][1]
        completion_token_ids = tokenizer.encode(completion)
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )

        if prompt_len < 2 or output_len < 2:
            continue

        if context_len and prompt_len + output_len > context_len:
            continue

        filtered_count += 1

    print(f"Total valid samples: {filtered_count}")

if __name__ == "__main__":
    # 修改为你的数据集路径和分词器名称
    dataset_path = "/workspace/datasets/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json"
    tokenizer_name_or_path = "/workspace/models/Llama-2-7b-hf"
    fixed_output_len = None
    context_len = None
    prompt_suffix = ""
    apply_chat_template = False

    filter_sharegpt_requests(
        dataset_path=dataset_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        fixed_output_len=fixed_output_len,
        context_len=context_len,
        prompt_suffix=prompt_suffix,
        apply_chat_template=apply_chat_template,
    )
    

"""
Results:

root@042d382ac496:/workspace/benchmark/lora# python sharegpt.py
Total valid samples: 92824
"""