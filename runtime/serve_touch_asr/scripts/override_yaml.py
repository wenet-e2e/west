"""YAML Override: 启动前覆盖配置文件中的 startup 参数。

- 支持分层格式（startup + runtime）
- CLI 接收 gpu-ids、tp-size、gpu-mem-util、port，写入 yaml startup 段
- shell 脚本在启动 server.py 前调用，确保 yaml 与实际运行参数一致
"""
import argparse
import sys

import yaml


def override_yaml(file_path, gpu_ids=None, tp_size=None, gpu_mem_util=None,
                  port=None):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

        if 'startup' not in config:
            config['startup'] = {}
        startup = config['startup']

        if gpu_ids is not None:
            startup['gpu_ids'] = gpu_ids
        if tp_size is not None:
            startup['tensor_parallel_size'] = int(tp_size)
        if gpu_mem_util is not None:
            startup['gpu_memory_utilization'] = float(gpu_mem_util)
        if port is not None:
            startup['port'] = int(port)

        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, sort_keys=False,
                      default_flow_style=False, allow_unicode=True)

    except Exception as e:
        print(f"Error updating YAML: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Override startup params in layered YAML config")
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--gpu-ids', type=str, default=None)
    parser.add_argument('--tp-size', type=int, default=None)
    parser.add_argument('--gpu-mem-util', type=float, default=None)
    parser.add_argument('--port', type=int, default=None)

    args = parser.parse_args()
    override_yaml(args.file, args.gpu_ids, args.tp_size, args.gpu_mem_util,
                  args.port)
