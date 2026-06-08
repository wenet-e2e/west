# Copyright (c) 2026 Pengshen Zhang
"""Engine: 进程级资源 + 启动配置 + 引擎运行时。

- runtime: ServiceRuntime / ServerSettings / EngineRuntime / EnginePhase
- loader: 模型加载、warmup 与 ready 状态
- config: YAML 热配置加载与校验
- env_snapshot: 启动阶段环境诊断
"""
