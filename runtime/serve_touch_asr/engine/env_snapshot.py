# Copyright (c) 2026 Pengshen Zhang
"""Startup Env Snapshot: 启动阶段环境诊断信息。

- collect_startup_snapshot: 汇总主机、进程、CUDA/NVIDIA 相关信息
- GpuSnapshot/StartupSnapshot: 结构化保存 GPU 和系统状态
- log_startup_snapshot: 将诊断信息写入启动日志，便于排查部署问题
"""
import logging
import os
import socket
import subprocess
from dataclasses import dataclass
from typing import List

try:
    import psutil
except ImportError:
    psutil = None

try:
    import pynvml
except ImportError:
    pynvml = None


@dataclass(frozen=True)
class StartupHostSnapshot:
    hostname: str
    ips: List[str]


@dataclass(frozen=True)
class StartupGpuSnapshot:
    index: str
    name: str
    total_mb: int
    free_mb: int
    used_mb: int


def collect_startup_host_snapshot() -> StartupHostSnapshot:
    hostname = socket.gethostname()
    ips: List[str] = []

    if psutil is not None:
        try:
            for addrs in psutil.net_if_addrs().values():
                for addr in addrs:
                    if (addr.family == socket.AF_INET
                            and not addr.address.startswith("127.")):
                        ips.append(addr.address)
        except Exception:
            pass

    if not ips:
        try:
            for item in socket.getaddrinfo(hostname, None, socket.AF_INET):
                ip = item[4][0]
                if not ip.startswith("127."):
                    ips.append(ip)
        except Exception:
            pass

    return StartupHostSnapshot(
        hostname=hostname,
        ips=sorted(set(ips)) or ["<unknown>"],
    )


def _collect_startup_gpu_snapshots_with_nvml() -> List[StartupGpuSnapshot]:
    if pynvml is None:
        raise RuntimeError("pynvml is not installed")

    pynvml.nvmlInit()
    try:
        gpus: List[StartupGpuSnapshot] = []
        for idx in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(StartupGpuSnapshot(
                index=str(idx),
                name=str(name),
                total_mb=int(mem.total // (1024 * 1024)),
                free_mb=int(mem.free // (1024 * 1024)),
                used_mb=int(mem.used // (1024 * 1024)),
            ))
        return gpus
    finally:
        pynvml.nvmlShutdown()


def _collect_startup_gpu_snapshots_with_nvidia_smi(
) -> List[StartupGpuSnapshot]:
    raw_gpu_rows = subprocess.check_output(
        ["nvidia-smi",
         "--query-gpu=index,name,memory.total,memory.free,"
         "memory.used",
         "--format=csv,noheader,nounits"],
        text=True
    ).strip().split('\n')

    gpus: List[StartupGpuSnapshot] = []
    for line in raw_gpu_rows:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 5:
            idx, name, total, free, used = parts[:5]
            gpus.append(StartupGpuSnapshot(
                index=idx,
                name=name,
                total_mb=int(total),
                free_mb=int(free),
                used_mb=int(used),
            ))
    return gpus


def collect_startup_gpu_snapshots() -> List[StartupGpuSnapshot]:
    try:
        return _collect_startup_gpu_snapshots_with_nvml()
    except Exception:
        pass
    try:
        return _collect_startup_gpu_snapshots_with_nvidia_smi()
    except Exception:
        return []


def log_startup_host_snapshot(logger: logging.Logger) -> None:
    """打印宿主机标识，便于事后从日志反查服务跑在哪台机器。"""
    try:
        snapshot = collect_startup_host_snapshot()
        logger.info(
            f"Host: {snapshot.hostname}  IPs: {', '.join(snapshot.ips)}")
    except Exception as e:
        logger.warning(f"Failed to resolve host/ip for logging: {e}")


def log_startup_gpu_snapshot(logger: logging.Logger) -> None:
    """打印当前 CUDA/GPU 状态。"""
    try:
        gpus = collect_startup_gpu_snapshots()

        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
        logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
        logger.info(
            f"CUDA_DEVICE_ORDER: "
            f"{os.environ.get('CUDA_DEVICE_ORDER', 'default')}")
        logger.info("GPU 状态:")
        if not gpus:
            logger.warning("未检测到可用 GPU 信息")
            return
        for gpu in gpus:
            logger.info(
                f"  GPU {gpu.index}: {gpu.name} | "
                f"Total: {gpu.total_mb}MB | "
                f"Free: {gpu.free_mb}MB | Used: {gpu.used_mb}MB")
            if cuda_visible != 'all':
                visible_gpus = [
                    g.strip() for g in cuda_visible.split(',')]
                if gpu.index in visible_gpus:
                    logger.info("    -> 将使用此 GPU")
    except Exception as e:
        logger.warning(f"无法获取 GPU 信息: {e}")
