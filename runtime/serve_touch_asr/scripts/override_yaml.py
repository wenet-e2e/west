import argparse
import sys

import yaml


def override_yaml(file_path, gpu_ids, tp_size, gpu_mem_util):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if 'stage_args' in config and isinstance(
                config['stage_args'], list) and len(
                config['stage_args']) > 0:
            stage_arg = config['stage_args'][0]

            # Update devices
            if 'runtime' not in stage_arg:
                stage_arg['runtime'] = {}
            stage_arg['runtime']['devices'] = gpu_ids

            # Update engine_args
            if 'engine_args' not in stage_arg:
                stage_arg['engine_args'] = {}
            stage_arg['engine_args']['tensor_parallel_size'] = int(tp_size)
            stage_arg['engine_args']['gpu_memory_utilization'] = float(
                gpu_mem_util)

            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    config,
                    f,
                    sort_keys=False,
                    default_flow_style=False,
                    allow_unicode=True)
            print(f"Successfully updated {file_path}")
        else:
            print(
                f"Error: Invalid YAML format in {file_path}. Could not find 'stage_args'.")
            sys.exit(1)

    except Exception as e:
        print(f"Error updating YAML file: {e}")
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Override YAML config for vLLM Server")
    parser.add_argument(
        '--file',
        type=str,
        required=True,
        help="Path to the YAML file")
    parser.add_argument(
        '--gpu-ids',
        type=str,
        required=True,
        help="GPU IDs (e.g., '0,1,2,3')")
    parser.add_argument(
        '--tp-size',
        type=int,
        required=True,
        help="Tensor Parallel Size")
    parser.add_argument(
        '--gpu-mem-util',
        type=float,
        required=True,
        help="GPU Memory Utilization")

    args = parser.parse_args()
    override_yaml(args.file, args.gpu_ids, args.tp_size, args.gpu_mem_util)
