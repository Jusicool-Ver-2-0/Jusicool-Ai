import psutil
import os
import time
import torch

# GPU 상태 체크 (선택적)
def print_gpu_usage():
    if torch.cuda.is_available():
        print("GPU 사용량:")
        print(os.popen("nvidia-smi").read())
    else:
        print("GPU를 사용할 수 없습니다.")

# CPU 및 메모리 사용량 체크
def print_cpu_memory_usage():
    cpu = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    print(f" CPU 사용량: {cpu}%")
    print(f"메모리 사용량: {memory.percent}% ({round(memory.used / (1024**3), 2)} GB / {round(memory.total / (1024**3), 2)} GB)")
