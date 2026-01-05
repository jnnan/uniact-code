import torch
import numpy as np
import os
import types
import transformers
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Qwen2_5_VLForConditionalGeneration
)
import time
import shutil

MOTION_TOKEN_CONFIG = {
    "start_id": 129625,
    "end_id": 129626,
    "code_base_id": 129627,
    "vocab_end_id": 151643
}

def load_finetuned_model(model_path):
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    if "qwen2_5" in config.model_type:
        model_class = Qwen2_5_VLForConditionalGeneration
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

    print(f"Loading base model (type: {config.model_type}) from {model_path}...")
    model = model_class.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    
    
    model.motion_token_start_id = MOTION_TOKEN_CONFIG['start_id']
    model.motion_token_end_id = MOTION_TOKEN_CONFIG['vocab_end_id']

    return model, tokenizer

def prepare_inference_input_t2m(tokenizer, description_part):
    im_start_token = "<|im_start|>"
    im_end_token = "<|im_end|>"
    text_part = tokenizer.encode(f"{im_start_token}user\nGenerate a motion code sequence for the following action: ", add_special_tokens=False)
    description_part = tokenizer.encode(description_part)
    assistant_part = tokenizer.encode(f"{im_end_token}\n{im_start_token}assistant\n", add_special_tokens=False)
    final_input_ids = text_part + description_part + assistant_part
    return torch.tensor([final_input_ids])

def create_motion_position_ids(input_ids_tensor, device):
    motion_start_token_id = MOTION_TOKEN_CONFIG['start_id']
    motion_end_token_id = MOTION_TOKEN_CONFIG['end_id']

    batch_size, seq_len = input_ids_tensor.shape
    final_pos_ids = torch.zeros_like(input_ids_tensor, dtype=torch.long)

    for i in range(batch_size):
        text_pos_counter = 0
        motion_pos_counter = 0
        in_motion = False

        for j in range(seq_len):
            token_id = input_ids_tensor[i, j]

            if in_motion:
                final_pos_ids[i, j] = motion_pos_counter
                motion_pos_counter += 1
                if token_id == motion_end_token_id:
                    in_motion = False
            else:
                final_pos_ids[i, j] = text_pos_counter
                text_pos_counter += 1
                if token_id == motion_start_token_id:
                    in_motion = True
                    motion_pos_counter = 0
    
    position_ids = (
        final_pos_ids
        .view(1, batch_size, seq_len)
        .expand(3, -1, -1)
    )
    
    return position_ids.to(device)

def parse_generated_ids(response_ids):
    """Parse motion codes from generated token IDs."""
    motion_codes = []
    for token_id in response_ids:
        if MOTION_TOKEN_CONFIG['code_base_id'] <= token_id <= MOTION_TOKEN_CONFIG['vocab_end_id']:
            motion_codes.append(token_id - MOTION_TOKEN_CONFIG['code_base_id'])
    return motion_codes

def encode_motion_tokens(motion_codes):
    """Encode motion code sequence to token ID sequence (reverse parsing)"""
    motion_token_ids = []
    for code in motion_codes:
        token_id = code + MOTION_TOKEN_CONFIG['code_base_id']
        motion_token_ids.append(token_id)
    return motion_token_ids

def unified_generation_step(model, tokenizer, prompt=None, prompt_length=None, motion_tokens=None, past_key_values=None, step_count=0):
    """
    Unified generation step function
    
    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        prompt: Text prompt (only used in first call)
        motion_tokens: Motion token sequence (only used in first call)
        past_key_values: Previous KV cache (used in subsequent calls)
        step_count: Current step count (for position encoding)
    
    Returns:
        next_token_id: Next token ID
        updated_past_key_values: Updated KV cache
        is_first_step: Whether this is the first step
    """
    device = model.device
    is_end_token = False
    
    #if prompt is not None and motion_tokens is not None:
    if prompt is not None:
        # First call: process prompt + motion tokens
        #print(f"  - First step: processing prompt + {len(motion_tokens)} motion tokens")
        
        # Prepare input
        input_ids = prepare_inference_input_t2m(tokenizer, prompt).to(device)
        start_token = torch.tensor([[MOTION_TOKEN_CONFIG['start_id']]], device=device)
        input_ids = torch.cat([input_ids, start_token], dim=1)
        
        # Add motion tokens
        if motion_tokens:
            motion_tensor = torch.tensor([motion_tokens], device=device)
            input_ids = torch.cat([input_ids, motion_tensor], dim=1)
        
        # Create position encoding
        position_ids = create_motion_position_ids(input_ids, device)
        prompt_length = input_ids.shape[1]
        
        # First forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=True,
                past_key_values=None,
                cache_position=torch.arange(prompt_length, device=device)
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values
        
        # Predict next token
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        if next_token_id.item() == MOTION_TOKEN_CONFIG['end_id']:
                next_token_id = torch.topk(next_token_logits, k=2, dim=-1)[1][..., 1].unsqueeze(-1)
                is_end_token = True
        
        return next_token_id, past_key_values, True, is_end_token, prompt_length
        
    else:
        # Subsequent calls: only process current token
        #print(f"  - Step {step_count}: generating next token")
        
        next_token_id = motion_tokens
        # Calculate position
        step = step_count-1
        next_position_ids = torch.tensor([[[step]]], device=device).expand(3, 1, 1)
        
        # Calculate global position
        cache_position = torch.tensor([prompt_length + step], device=device)
        
        # Generate next token
        with torch.no_grad():
            outputs = model(
                input_ids=next_token_id,
                position_ids=next_position_ids,
                use_cache=True,
                past_key_values=past_key_values,
                cache_position=cache_position
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values
        
        # Predict next token
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        if next_token_id.item() == MOTION_TOKEN_CONFIG['end_id']:
                next_token_id = torch.topk(next_token_logits, k=2, dim=-1)[1][..., 1].unsqueeze(-1)
                is_end_token = True
        return next_token_id, past_key_values, False, is_end_token, prompt_length