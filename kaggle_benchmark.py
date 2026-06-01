# ==============================================================================
# Open-Source Telemetry Benchmark Harness (Public Baseline Version)
# ==============================================================================
import os
import sys
import time
import torch
from transformers import AutoTokenizer
import psutil

# Disable HF warnings and caches if needed
os.environ['HF_HOME'] = '/kaggle/working/huggingface_cache'
os.environ['HF_HUB_DISABLE_XET'] = '1'

# ==============================================================================
# LAYER-STREAMING ENGINE PUBLIC INTERFACE
# ==============================================================================
class LayerStreamingRunner:
    def __init__(self, input_dir="/kaggle/input"):
        self.input_dir = input_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Initializing LayerStreamingRunner on: {self.device}")
        
        # The public version runs the open-source baseline Qwen model.
        # The high-performance proprietary DeepSeek layer-streaming engine is withheld.
        self.use_fallback = True
        
        online_model_id = "Qwen/Qwen1.5-0.5B-Chat"
        print(f"[INFO] Initializing baseline model from Hugging Face Hub: {online_model_id}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(online_model_id)
            from transformers import AutoModelForCausalLM
            self.fallback_model = AutoModelForCausalLM.from_pretrained(online_model_id).to(self.device)
            print("[SUCCESS] Loaded baseline model successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to load baseline model: {e}")
            self.fallback_model = None
            self.tokenizer = None

    def generate(self, prompt: str, max_new_tokens: int = 15, temperature: float = 0.7) -> str:
        if self.tokenizer is None or self.fallback_model is None:
            return "[ERROR] Cannot generate: Baseline model is not initialized."
            
        # Format the prompt using model's chat template if available to get coherent outputs
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.fallback_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
        # Slice output to decode only the newly generated tokens
        decoded = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return decoded

# ==============================================================================
# TELEMETRY HARNESS & EXECUTOR
# ==============================================================================
def measure_footprint():
    vram_val = 0.0
    if torch.cuda.is_available():
        vram_val = torch.cuda.max_memory_allocated() / (1024 ** 3)
    process = psutil.Process(os.getpid())
    ram_val = process.memory_info().rss / (1024 ** 3)
    return vram_val, ram_val

def run_evaluation_benchmark():
    print("=" * 70)
    print("      DEEPSEEK 4-BIT LAYER-STREAMING TELEMETRY BENCHMARK HARNESS      ")
    print("=" * 70)
    
    t_init_start = time.time()
    runner = LayerStreamingRunner(input_dir="/kaggle/input")
    t_init_end = time.time()
    
    load_vram, load_ram = measure_footprint()
    
    print("\n" + "-" * 70)
    print(f"[*] Engine Loader binding duration: {t_init_end - t_init_start:.2f} seconds")
    print(f"[*] Base Engine memory usage (VRAM): {load_vram:.2f} GB")
    print(f"[*] Base Engine memory usage (RAM) : {load_ram:.2f} GB")
    print("-" * 70 + "\n")
    
    test_prompts = [
        "Create a clean python matrix transformation function to flip a grid 90 degrees.",
        "System diagnostics verification loop initialized. Status parameters:",
        "Solve ARC grid: 3x3 pattern input. Target color change sequence:"
    ]
    
    report_lines = [
        "# DeepSeek Layer-Streaming Engine Telemetry Report",
        f"- **Device Used**: {runner.device}",
        f"- **Engine Load Time**: {t_init_end - t_init_start:.2f}s",
        f"- **Fallback Active**: {runner.use_fallback}",
        "",
        "| Test Prompt ID | Output Latency (s) | Generation Speed (tokens/s) | Peak VRAM (GB) | Peak System RAM (GB) |",
        "|---|---|---|---|---|---|"
    ]
    
    for idx, prompt in enumerate(test_prompts):
        print(f"[*] Executing Test prompt #{idx+1}: '{prompt}'")
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        t_gen_start = time.time()
        max_new_tokens = 25
        output = runner.generate(prompt, max_new_tokens=max_new_tokens)
        t_gen_end = time.time()
        
        gen_vram, gen_ram = measure_footprint()
        duration = t_gen_end - t_gen_start
        throughput = max_new_tokens / duration if duration > 0 else 0
        
        print(f"    -> Output latency: {duration:.2f}s | Throughput: {throughput:.1f} tokens/sec")
        print(f"    -> Peak VRAM: {gen_vram:.2f} GB | Peak System RAM: {gen_ram:.2f} GB")
        print(f"    -> Generated Text: {output}\n")
        
        report_lines.append(f"| Prompt {idx+1} | {duration:.2f}s | {throughput:.1f} t/s | {gen_vram:.2f} GB | {gen_ram:.2f} GB |")

    # Save Markdown report log for proving invention metrics
    model_name = "qwen_fallback" if runner.use_fallback else "deepseek_v4_flash"
    report_content = "\n".join(report_lines)
    report_path = f"/kaggle/working/benchmark_report_{model_name}.md"
    try:
        with open(report_path, "w") as f:
            f.write(report_content)
        print(f"[SUCCESS] Telemetry report saved to: {report_path}")
    except Exception as e:
        print(f"[WARNING] Failed to write report file: {e}")
        
    print("=" * 70)
    print("                         BENCHMARK COMPLETED                          ")
    print("=" * 70)

if __name__ == "__main__":
    run_evaluation_benchmark()
