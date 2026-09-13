import os
import json
import time
import platform
import subprocess
import psutil

# torch is optional: the bulk DP-NMF generator is sklearn-only and runs on CPU.
# Fall back to a stub exposing torch.cuda.is_available() == False so the profiler
# transparently uses RAM-based (psutil) measurement when torch isn't installed.
try:
    import torch
except ImportError:  # pragma: no cover - CPU-only environments
    class _NoCuda:
        @staticmethod
        def is_available():
            return False
    class _TorchStub:
        cuda = _NoCuda()
    torch = _TorchStub()


class ResourceProfiler:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.start_time = None

    def start(self):
        self.start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def stop(self):
        if self.start_time is None:
            raise RuntimeError("Profiler not started")

        elapsed = time.time() - self.start_time
        return {
            "elapsed_sec": elapsed,
            **self.get_peak_memory(),
            **self.get_gpu_info()
        }

    def get_gpu_info(self):
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            return {
                "gpu_name": props.name,
                "gpu_total_memory_gb": round(props.total_memory / (1024**3), 3),
                "gpu_index": idx
            }
        return {"gpu_name": "CPU", "gpu_total_memory_gb": None}

    def get_peak_memory(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return {
                "peak_memory_gb": round(
                    torch.cuda.max_memory_allocated() / (1024**3), 3
                ),
                "memory_type": "GPU"
            }

        proc = psutil.Process(os.getpid())
        return {
            "peak_memory_gb": round(proc.memory_info().rss / (1024**3), 3),
            "memory_type": "RAM"
        }

    def get_cpu_info(self):
        """CPU/platform details (arch, model, core counts, OS)."""
        model = platform.processor()
        try:
            if platform.system() == "Darwin":
                model = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            elif platform.system() == "Linux":
                for line in open("/proc/cpuinfo"):
                    if line.lower().startswith("model name"):
                        model = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
        return {
            "cpu_arch": platform.machine(),                 # e.g. x86_64
            "cpu_model": model,
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "os": f"{platform.system()} {platform.release()}",
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        }

    def count_parameters(self, model):
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return round(total / 1e6, 3)

    def save_metrics(self, metrics, split_no, experiment_name):
        out_dir = "resource_logs"
        os.makedirs(out_dir, exist_ok=True)

        fname = f"{out_dir}/{experiment_name or 'run'}_split{split_no}.json"

        with open(fname, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"[resource] saved -> {fname}")