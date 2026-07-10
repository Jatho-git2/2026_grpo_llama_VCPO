#!/usr/bin/env python
# coding: utf-8

# # Notebook #2: Training
# 
# ## VCPO on Llama Code Demo
# 
# In this notebook, we'll:
# 
# * Load in the baseline ~8B parameter Llama 3.1 model
# * Load in the GSM8K dataset
# * Add LoRa adapters to the model
# * Set up GRPO training for this model and dataset, including specifying the reward functions
# * Add a callback to training that saves the checkpointed model every 25 training steps
# * Also log the rewards in the notebook every 25 steps
# * Train the model, optionally restarting training from one of the checkpoints

# # Imports, installation, and setup

# In[1]:


# this cell should take ~15 seconds
import os
# Force PyTorch to use the exact same numbering as nvidia-smi
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# Now select GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
print(f"Active GPU: {torch.cuda.get_device_name(0)}")

from datetime import datetime
import json
from os import path
import random
import re
import shutil
import typing

import datasets
import huggingface_hub
import peft
from peft import TaskType
import torch
from torch import cuda
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import trl


# In[2]:


# authentication required to read in the Llama model
huggingface_hub.login()


# In[3]:


if not path.exists('data'):
    os.mkdir("data")


# # IO functions

# In[4]:


JSONable = typing.Union[dict, list]

def save_jsonable_object_locally(obj: JSONable, filename: str, save_dir: str = "data") -> None:
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"{filename}.json")
    print(f"Saving object locally to {filepath}")
    with open(filepath, "w") as f:
        json.dump(obj, f, indent=2)

def load_jsonable_object_locally(filename: str, load_dir: str = "data") -> JSONable:
    filepath = os.path.join(load_dir, f"{filename}.json")
    print(f"Reading {filepath} into Python")
    with open(filepath, "r") as f:
        return json.load(f)


# # Loading in the model and applying LoRa weights

# In[6]:


import peft
from peft import TaskType
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_SEQ_LENGTH = 256
LORA_RANK = 64
MODEL_NAME = "meta-llama/meta-Llama-3.1-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Apply LoRA with PEFT
lora_config = peft.LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_RANK,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)
model = peft.get_peft_model(model, lora_config)


# Code to see total number of parameters:
# 
# ```python
# # Count up total parameters
# total_params = sum(p.numel() for p in model.parameters())
# 
# print(f"Total params: {total_params:,}")
# ```
# 
# ```
# Total params: 8,198,033,408
# ```

# Code to see total number of parameters:
# 
# ```python
# # Count up trainable vs. total parameters
# total_params = sum(p.numel() for p in model.parameters())
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# 
# print(f"Total params:      {total_params:,}")
# print(
#     f"Trainable params:  {trainable_params:,} "
#     f"({trainable_params/total_params*100:.4f}% of total)"
# )
# ```
# 
# ```
# Total params:      8,198,033,408
# Trainable params:  167,772,160 (2.0465% of total)
# ```

# In[7]:


SYSTEM_PROMPT = """
### EXAMPLE ###
Q: 3+2
<reasoning>
3 plus 2 is 5
</reasoning>
<answer>
5
</answer>

Now follow the same format EXACTLY for each question:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def _extract_hash_answer(text: str) -> str:
    return text.split("####")[1].strip()

# from 
# https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb
def _extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()
    

def _parse_number(raw: str) -> typing.Optional[float]:
    raw = raw.strip().strip("$").replace(",", "")
    # remove trailing period, etc.
    raw = re.sub(r"[.!]+$", "", raw)
    try:
        return float(raw)
    except ValueError:
        return None



# from 
# https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb
def _get_gsm8k_questions(split = "train") -> datasets.Dataset:
    data = datasets.load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(
        lambda x: {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['question']},
            ],
            'answer': _extract_hash_answer(x['answer']),
        }
    )
    return data # type: ignore

gsm8k_dataset = _get_gsm8k_questions()



def _correctness_reward_func_length_penalty(
    completions: list[list[dict[str, str]]], answer: list[str], **kwargs
) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    predicted_nums = [_parse_number(_extract_xml_answer(r)) for r in responses]
    gold_nums = [_parse_number(a) for a in answer]
    rewards = []
    for r, p, g in zip(responses, predicted_nums, gold_nums):
        if p is not None and g is not None and abs(p - g) < 1e-9:  # Answer is correct
            try:
                reasoning_text = r.split("<reasoning>")[1].split("</reasoning>")[0].strip()
                reasoning_length = len(reasoning_text)
                if reasoning_length < 45:
                    return 0
                length_factor = min(1.0, reasoning_length / 100.0)  # Scale up to 100 characters
                reward = 5.0 * length_factor
            except IndexError:
                reward = 0.0  # Incorrect format prevents reward
        else:
            reward = 0.0  # Incorrect answer
        rewards.append(reward)
    return rewards

# from 
# https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb
def _int_reward_func(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [_extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def _format_reward_func(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    pattern = r"^[\s]*<reasoning>[\s\S]+?</reasoning>\s*<answer>[\s\S]+?</answer>[\s]*$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [1 if match else 0.0 for match in matches]


# In[8]:


class CheckpointCallback(transformers.TrainerCallback):
    def __init__(
        self, steps_interval=10, output_dir="./data/2-Training", local_prefix="lora_checkpoints"
    ):
        self.steps_interval = steps_interval
        self.output_dir = output_dir
        self.local_prefix = local_prefix

    def on_step_end(self, args, state, control, **kwargs):
        """
        Every steps_interval steps, we request the trainer to perform a checkpoint save.
        This triggers the on_save event below.
        """
        if state.global_step > 0 and (state.global_step % self.steps_interval == 0):
            control.should_save = True  # signals the trainer to save now

    # def on_save(self, args, state, control, **kwargs):
    #     """
    #     Called once the trainer has actually saved the full checkpoint to
    #     `outputs/checkpoint-<step>`.

    #     We save that folder .
    #     """
    #     checkpoint_dir = path.join(self.output_dir, f"checkpoint-{state.global_step}")
    #     if not path.exists(checkpoint_dir):
    #         print(
    #             "S3CheckpointCallback: no checkpoint directory found at {checkpoint_dir}, "
    #             "skipping."
    #         )
    #         return

    #     # Zip
    #     zip_base = f"checkpoint_{state.global_step}"
    #     zip_filename = f"{zip_base}.zip"
    #     shutil.make_archive(base_name=zip_base, format="zip", root_dir=checkpoint_dir)
    #     print(f"S3CheckpointCallback: Created {zip_filename} from {checkpoint_dir}")

    #     # Upload
    #     s3_key = f"{self.s3_prefix}/{zip_filename}"
    #     upload_file_to_s3(zip_filename, s3_key)

    #     os.remove(zip_filename)
    #     print(
    #         f"S3CheckpointCallback: checkpoint {state.global_step} zipped+uploaded to "
    #         "s3://{S3_BUCKET_NAME}/{s3_key}"
    #     )


class LogRewardsGRPOTrainer(trl.GRPOTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Compute the loss as usual
        loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        # Log rewards and sub-rewards from self._metrics
        mode = "train"  # Since we're in training
        # Align with logging_steps
        if (
            self.state.global_step
            and self.state.global_step % self.args.logging_steps == 0
        ):  
            logs = {}
            # Overall reward and reward_std
            if self._metrics[mode]["reward"]:
                logs["reward"] = self._metrics[mode]["reward"][-1]  # Most recent value
            if self._metrics[mode]["reward_std"]:
                logs["reward_std"] = self._metrics[mode]["reward_std"][-1]
            # Sub-rewards
            for i, reward_func in enumerate(self.reward_funcs):
                reward_func_name = reward_func.__name__
                metric_key = f"rewards/{reward_func_name}"
                if self._metrics[mode][metric_key]:
                    logs[f"subreward_{reward_func_name}"] = self._metrics[mode][
                        metric_key
                    ][-1]
            # Additional metrics like completion_length and kl (if applicable)
            if self._metrics[mode]["completion_length"]:
                logs["completion_length"] = self._metrics[mode]["completion_length"][-1]
            if self._metrics[mode].get("kl"):
                logs["kl"] = self._metrics[mode]["kl"][-1]
            if logs:
                self.log(logs)

            print(logs)
        return loss


# ## Optional code to restart training from a checkpoint

# In[9]:


def _resume_from_latest_local_checkpoint(
    local_dir="data/2-Training"):
    """
    Finds the highest-numbered checkpoint_XX.json in local directory, returns that path.
    """

    # Check to see if the directory is real
    if not path.exists(local_dir):
        return None
    
    # Get all the relevant checkpoint directories into a list, otherwise if none, 
    # return none.
    checkpoints = [d for d in os.listdir(local_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    
    # get the highest numbered (best) checkpoint
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))

    # return the full path
    return path.join(local_dir, latest_checkpoint)


# # Training

# In[11]:


are_we_at_the_start_of_the_training_run = False

training_args = trl.GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_torch",
    logging_steps=25,
    bf16=cuda.is_bf16_supported(),
    fp16=not cuda.is_bf16_supported(),
    per_device_train_batch_size=6,
    gradient_accumulation_steps=4,
    num_generations=6,
    #max_prompt_length=256,
    max_completion_length=256,
    max_steps=2_500,
    save_strategy="steps",
    save_steps=-1,
    max_grad_norm=0.1,
    report_to="none",
    output_dir="data/2-Training",
    seed=250217,
    gradient_checkpointing=True,
)

# Add custom callback to do the custom saving
checkpoint_callback = CheckpointCallback(
    steps_interval=25, output_dir="data/2-Training", local_prefix="lora_checkpoints"
)

trainer = LogRewardsGRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        _correctness_reward_func_length_penalty,
        _format_reward_func,
        _int_reward_func,
    ],
    args=training_args,
    train_dataset=gsm8k_dataset,
)

trainer.add_callback(checkpoint_callback)

if not are_we_at_the_start_of_the_training_run:
    trainer.train(_resume_from_latest_local_checkpoint(local_dir="data/2-Training"))
else:
    trainer.train()


# Due to the `LogRewardsGRPOTrainer`, this will print out lines like the following (these are the actual lines from step 400 of the training run):
# 
# ```
# {'reward': 3.1666667461395264, 'reward_std': 2.58198881149292, 'subreward_correctness_reward_func_length_penalty': 1.6666666269302368, 'subreward_format_reward_func': 1.0, 'subreward_int_reward_func': 0.5, 'completion_length': 157.1666717529297, 'kl': 0.006420954596251249}
# {'reward': 3.1666667461395264, 'reward_std': 2.58198881149292, 'subreward_correctness_reward_func_length_penalty': 1.6666666269302368, 'subreward_format_reward_func': 1.0, 'subreward_int_reward_func': 0.5, 'completion_length': 144.83334350585938, 'kl': 0.008445960469543934}
# {'reward': 3.1666667461395264, 'reward_std': 2.58198881149292, 'subreward_correctness_reward_func_length_penalty': 1.6666666269302368, 'subreward_format_reward_func': 1.0, 'subreward_int_reward_func': 0.5, 'completion_length': 132.5, 'kl': 0.006176165770739317}
# {'reward': 6.5, 'reward_std': 0.0, 'subreward_correctness_reward_func_length_penalty': 5.0, 'subreward_format_reward_func': 1.0, 'subreward_int_reward_func': 0.5, 'completion_length': 134.33334350585938, 'kl': 0.005382485222071409}
# ```
# 
# in addition to the standard:
# 
# ```
# Step	Training Loss
# 25	0.044700
# 50	0.043200
# 75	0.040600
# 100	0.028800
# 125	0.021200
# 150	0.048000
# 175	0.035600
# 200	0.016900
# 225	0.016000
# 250	0.014200
# 275	0.015000
# 300	0.022900
# 325	0.012700
# 350	0.026400
# 375	0.028500
# 400	0.015500
# ```
# 
# that is printed. These `Training Loss`es are plotted below:
# 
# ![Proportion Plot](https://data-science-talks.s3.us-east-1.amazonaws.com/odsc_east_2025/images/training_loss_plot.png)
