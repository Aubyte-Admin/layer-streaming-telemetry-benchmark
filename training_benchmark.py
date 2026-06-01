# ==============================================================================
# Open-Source Telemetry Benchmark Harness for Training Affordability
# ==============================================================================
import os
import sys
import gc
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil

# Disable HF warnings and caches if needed
os.environ['HF_HOME'] = '/kaggle/working/huggingface_cache'
os.environ['HF_HUB_DISABLE_XET'] = '1'

def measure_footprint():
    vram_val = 0.0
    if torch.cuda.is_available():
        vram_val = torch.cuda.max_memory_allocated() / (1024 ** 3)
    process = psutil.Process(os.getpid())
    ram_val = process.memory_info().rss / (1024 ** 3)
    return vram_val, ram_val

def run_training_step(model, inputs, optimizer, grad_checkpoint=False):
    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    if grad_checkpoint:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
        
    start_time = time.time()
    
    # Forward pass
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    
    # Backward pass
    loss.backward()
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    duration = time.time() - start_time
    
    vram_val, ram_val = measure_footprint()
    return vram_val, ram_val, duration, loss.item()

def run_training_benchmark():
    print("=" * 75)
    print("       OPEN-SOURCE TRAINING AFFORDABILITY TELEMETRY BENCHMARK        ")
    print("=" * 75)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Initializing on device: {device}")
    
    online_model_id = "Qwen/Qwen1.5-0.5B-Chat"
    print(f"[INFO] Loading baseline model from Hugging Face Hub: {online_model_id}...")
    
    t_init_start = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(online_model_id)
        model = AutoModelForCausalLM.from_pretrained(online_model_id).to(device)
        print("[SUCCESS] Loaded model successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    t_init_end = time.time()
    
    init_vram, init_ram = measure_footprint()
    print(f"[INFO] Model loading completed in {t_init_end - t_init_start:.2f}s")
    print(f"[INFO] Base Engine Memory (VRAM): {init_vram:.2f} GB | (Host RAM): {init_ram:.2f} GB\n")
    
    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Dummy text to train on
    prompt_text = "Standard layer-streaming configuration running validation loop with gradient accumulation to demonstrate train memory efficiency."
    
    # Matrix of experiments to run
    # Format: (Batch Size, Gradient Checkpointing, Name)
    experiments = [
        (1, False, "Micro-Batch (BS=1) | Standard Training"),
        (1, True,  "Micro-Batch (BS=1) | Gradient Checkpointing (APS)"),
        (4, False, "Standard Batch (BS=4) | Standard Training"),
        (4, True,  "Standard Batch (BS=4) | Gradient Checkpointing (APS)"),
    ]
    
    report_lines = [
        "# Training Affordability Telemetry Report",
        f"- **Device Used**: {device}",
        f"- **Base Model**: {online_model_id}",
        f"- **Model Parameters**: ~620M",
        "",
        "| Experiment Configuration | Peak VRAM (GB) | Peak System RAM (GB) | Step Latency (s) | Loss Value |",
        "|---|---|---|---|---|",
    ]
    
    # Run warmup step to initialize gradients and optimizer states
    print("[*] Running warmup step...")
    warmup_inputs = tokenizer([prompt_text], return_tensors="pt", padding=True).to(device)
    _ = run_training_step(model, warmup_inputs, optimizer, grad_checkpoint=False)
    print("[SUCCESS] Warmup complete.\n")
    
    for bs, grad_chk, name in experiments:
        print(f"[*] Benchmarking: {name} ...")
        # Prepare inputs with specific batch size
        batch_texts = [f"{prompt_text} Index: {i}" for i in range(bs)]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to(device)
        
        # Measure peak memory and speed
        vram, ram, duration, loss = run_training_step(model, inputs, optimizer, grad_chk)
        
        # For CPU-only runs, VRAM will be 0.00, we'll represent Host RAM clearly
        print(f"    -> Step latency: {duration:.2f}s | Loss: {loss:.4f}")
        print(f"    -> Peak VRAM: {vram:.2f} GB | Peak System RAM: {ram:.2f} GB\n")
        
        report_lines.append(f"| {name} | {vram:.2f} GB | {ram:.2f} GB | {duration:.2f}s | {loss:.4f} |")
        
    report_content = "\n".join(report_lines)
    report_path = "/kaggle/working/training_telemetry_report.md" if os.path.exists("/kaggle/working") else "./training_telemetry_report.md"
    
    try:
        with open(report_path, "w") as f:
            f.write(report_content)
        print(f"[SUCCESS] Telemetry report saved to: {report_path}")
    except Exception as e:
        print(f"[WARNING] Failed to write report file: {e}")
        
    print("=" * 75)
    print("                       BENCHMARK RUN COMPLETED                        ")
    print("=" * 75)

if __name__ == "__main__":
    run_training_benchmark()
