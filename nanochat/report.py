"""
Utilities for generating pretraining report cards.
"""

import argparse
import datetime
import os
import platform
import socket
import subprocess

import psutil
import torch


def run_command(cmd):
    """Run a shell command and return output, or None if it fails."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        if result.stdout.strip():
            return result.stdout.strip()
        if result.returncode == 0:
            return ""
        return None
    except Exception:
        return None


def get_git_info():
    info = {}
    info["commit"] = run_command("git rev-parse --short HEAD") or "unknown"
    info["branch"] = run_command("git rev-parse --abbrev-ref HEAD") or "unknown"
    status = run_command("git status --porcelain")
    info["dirty"] = bool(status) if status is not None else False
    info["message"] = run_command("git log -1 --pretty=%B") or ""
    info["message"] = info["message"].split("\n")[0][:80]
    return info


def get_gpu_info():
    if not torch.cuda.is_available():
        return {"available": False}

    num_devices = torch.cuda.device_count()
    info = {
        "available": True,
        "count": num_devices,
        "names": [],
        "memory_gb": [],
    }
    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        info["names"].append(props.name)
        info["memory_gb"].append(props.total_memory / (1024**3))
    info["cuda_version"] = torch.version.cuda or "unknown"
    return info


def get_system_info():
    return {
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "user": os.environ.get("USER", "unknown"),
        "base_dir": os.environ.get("NANOCHAT_BASE_DIR", ".exps/nanochat"),
        "working_dir": os.getcwd(),
    }




def generate_header():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    git_info = get_git_info()
    gpu_info = get_gpu_info()
    sys_info = get_system_info()

    header = f"""# Pretraining Report

Generated: {timestamp}

## Environment

### Git Information
- Branch: {git_info['branch']}
- Commit: {git_info['commit']} {"(dirty)" if git_info['dirty'] else "(clean)"}
- Message: {git_info['message']}

### Hardware
- Platform: {sys_info['platform']}
- CPUs: {sys_info['cpu_count']} cores ({sys_info['cpu_count_logical']} logical)
- Memory: {sys_info['memory_gb']:.1f} GB
"""

    if gpu_info.get("available"):
        gpu_names = ", ".join(set(gpu_info["names"]))
        total_vram = sum(gpu_info["memory_gb"])
        header += f"""- GPUs: {gpu_info['count']}x {gpu_names}
- GPU Memory: {total_vram:.1f} GB total
- CUDA Version: {gpu_info['cuda_version']}
"""
    else:
        header += "- GPUs: None available\n"

    header += f"- Hourly Rate: ${cost_info['hourly_rate']:.2f}/hour\n"

    header += f"""
### Software
- Python: {sys_info['python_version']}
- PyTorch: {sys_info['torch_version']}

"""

    return header


def slugify(text):
    return text.lower().replace(" ", "-")


EXPECTED_FILES = [
    "tokenizer-training.md",
    "base-model-training.md",
    "base-model-evaluation.md",
    "base-model-multi-seed-summary.md",
]


def extract(section, keys):
    if not isinstance(keys, list):
        keys = [keys]
    out = {}
    for line in section.split("\n"):
        for key in keys:
            if key in line:
                out[key] = line.split(":", 1)[1].strip()
    return out


def extract_timestamp(content, prefix):
    for line in content.split("\n"):
        if line.startswith(prefix):
            time_str = line.split(":", 1)[1].strip()
            try:
                return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
    return None


class Report:
    def __init__(self, report_dir):
        os.makedirs(report_dir, exist_ok=True)
        self.report_dir = report_dir

    def log(self, section, data):
        slug = slugify(section)
        file_name = f"{slug}.md"
        file_path = os.path.join(self.report_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"## {section}\n")
            f.write(f"timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for item in data:
                if not item:
                    continue
                if isinstance(item, str):
                    f.write(item)
                else:
                    for k, v in item.items():
                        if isinstance(v, float):
                            vstr = f"{v:.4f}"
                        elif isinstance(v, int) and v >= 10000:
                            vstr = f"{v:,.0f}"
                        else:
                            vstr = str(v)
                        f.write(f"- {k}: {vstr}\n")
            f.write("\n")
        return file_path

    def generate(self):
        report_file = os.path.join(self.report_dir, "report.md")
        print(f"Generating report to {report_file}")
        final_metrics = {}
        start_time = None
        end_time = None
        with open(report_file, "w", encoding="utf-8") as out_file:
            header_file = os.path.join(self.report_dir, "header.md")
            if os.path.exists(header_file):
                with open(header_file, "r", encoding="utf-8") as f:
                    header_content = f.read()
                out_file.write(header_content)
                start_time = extract_timestamp(header_content, "Run started:")
            else:
                print(f"Warning: {header_file} does not exist. Did you forget to run `nanochat.report reset`?")

            for file_name in EXPECTED_FILES:
                section_file = os.path.join(self.report_dir, file_name)
                if not os.path.exists(section_file):
                    continue
                with open(section_file, "r", encoding="utf-8") as in_file:
                    section = in_file.read()
                end_time = extract_timestamp(section, "timestamp:")
                if file_name == "base-model-evaluation.md":
                    final_metrics = extract(section, ["CORE metric", "train bpb", "val bpb"])
                out_file.write(section)
                out_file.write("\n")

            out_file.write("## Summary\n\n")
            if final_metrics:
                for key in ["CORE metric", "train bpb", "val bpb"]:
                    if key in final_metrics:
                        out_file.write(f"- {key}: {final_metrics[key]}\n")
                out_file.write("\n")

            if start_time and end_time:
                duration = end_time - start_time
                total_seconds = int(duration.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                out_file.write(f"Total wall clock time: {hours}h{minutes}m\n")
            else:
                out_file.write("Total wall clock time: unknown\n")
        return report_file

    def reset(self):
        for file_name in EXPECTED_FILES:
            file_path = os.path.join(self.report_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
        report_file = os.path.join(self.report_dir, "report.md")
        if os.path.exists(report_file):
            os.remove(report_file)
        header_file = os.path.join(self.report_dir, "header.md")
        header = generate_header()
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(header_file, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(f"Run started: {start_time}\n\n---\n\n")
        print(f"Reset report and wrote header to {header_file}")


class DummyReport:
    def log(self, *args, **kwargs):
        pass

    def reset(self, *args, **kwargs):
        pass

    def generate(self, *args, **kwargs):
        pass


def _resolve_report_dir(model_tag=None):
    from nanochat.common import get_base_dir

    resolved_tag = model_tag or os.environ.get("MODEL_TAG") or "default"
    base_dir = get_base_dir()
    return os.path.join(base_dir, "report", resolved_tag)


def get_report(model_tag=None):
    from nanochat.common import get_dist_info

    _, ddp_rank, _, _ = get_dist_info()
    if ddp_rank == 0:
        return Report(_resolve_report_dir(model_tag=model_tag))
    return DummyReport()


def main():
    parser = argparse.ArgumentParser(description="Pretraining report helper")
    parser.add_argument("action", choices=["generate", "reset"], help="report action to run")
    parser.add_argument("--model-tag", type=str, default=None, help="report model tag")
    args = parser.parse_args()

    report = get_report(model_tag=args.model_tag)
    if args.action == "generate":
        report.generate()
    else:
        report.reset()


if __name__ == "__main__":
    main()
