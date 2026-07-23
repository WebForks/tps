# TPS Calculator 功能说明

**最后更新**: 2026-07-23
**模型数量**: 368 个
**GPU 数量**: 250 个

---

## 计算输入

| 参数 | 说明 |
|------|------|
| GPU 型号 | 支持 NVIDIA / AMD / Intel / Apple / 国产芯片 |
| GPU 数量 | 预设 1/2/4/8/16，支持自定义（1-512）；多卡 TP 时显存按**每卡**展示 |
| 互联方式 | NVLink / InfiniBand / PCIe |
| 模型 | 368 个规范 Dense/MoE 模型，含 VLM |
| 量化精度 | FP32 / BF16 / FP8 / INT8 / INT4 / GGUF 各档位（Apple/llama.cpp 自动用 `gguf_bytes`） |
| 上下文长度 | 任意设置 |
| 并发请求数 | 预设 + 自定义 |
| Prompt / 输出长度 | 影响 KV Cache 和延迟估算 |
| 推理框架 | vLLM / TRT-LLM / SGLang / LMDeploy / TGI / llama.cpp / ExLlamaV2 / MLX / 理论上限 |
| 计算模式 | GPU、GPU + CPU Offload、纯 CPU |
| 系统内存 | DDR3 / DDR4 / DDR5、传输速率（MT/s）、通道数、可选实测带宽和已安装容量（GB） |

**高级选项**:
- Flash Attention 开关
- KV Cache 量化（按所选框架过滤其实际支持的格式）
- Prefix Cache 命中率
- MoE CPU Offload + PCIe 代际/链路宽度
- llama.cpp 分层 GPU Offload（`--n-gpu-layers`）
- Speculative Decoding（接受率 + draft 长度 + draft 模型参数量）
- Pipeline Parallel 阶段数
- Expert Parallel 并行度
- 图像数量（VLM 模型，消耗上下文并影响 Prefill/Decode）

**系统内存带宽**：按 `MT/s × 8 bytes × 通道数 ÷ 1000` 计算理论 GB/s；如果提供实测带宽，则直接使用实测值。框架效率单独应用，不会对带宽重复折损。默认值仍为双通道 DDR5-4800（76.8 GB/s）。

**CPU 算力**：纯 CPU 与 llama.cpp 分层 Offload 可选填峰值 FP16/BF16 TFLOPS；所选框架的效率系数随后单独应用，用于约束 Prefill 和 compute-bound Decode。留空时不会虚构桌面 CPU 算力，CPU 速度会标记为仅受内存带宽约束的上界。

**Apple 默认**：选择 Apple Silicon 时框架默认 **MLX**（非 llama.cpp metal）。

---

## 计算输出

| 输出 | 说明 |
|------|------|
| 显存占用 | 权重 / KV Cache / 激活值 / 系统开销；多卡 TP 显示**每卡需求**与集群合计 |
| 可运行判断 | 同时校验每卡显存、系统 RAM（预留 10%）、上下文、并行拓扑、框架/硬件和 KV 格式兼容性 |
| Decode 吞吐 | 总 tok/s（含上下界区间） |
| 单请求速度 | 中位 tok/s，1 位小数 |
| Prefill 吞吐 | 算力上限和实际估算 |
| TTFT | 首 token 延迟（ms） |
| TPOT | 单 token 延迟（ms） |
| 总延迟 | `TTFT + (输出长度 - 1) × TPOT`；TTFT 已包含第一个输出 token |
| 瓶颈判断 | 带宽瓶颈 / 算力瓶颈 |
| Roofline 比 | 带宽上限 vs 算力上限 |
| TP 通信效率 | 多卡时 all-reduce 效率损耗 |
| PP 气泡效率 | Pipeline Parallel 流水线填充效率 |
| 功耗估算 | GPU 总 TDP（kW） |
| 精度评级 | 按模型结构、硬件拓扑、Offload、Speculative/VLM 等近似程度标注 high / mid / low |

---

## 校准与回归

估算器使用公开 benchmark 做回归检查，同时强制遵守权重流量、KV 流量、计算峰值和通信带宽等物理上限。它是容量规划模型，不是对特定驱动、内核、模型文件和服务器负载的实测替代品。

| 平台 | 主要修正项 |
|------|-----------|
| Apple MLX | `decodeBwScale` 按代际/SKU、统一内存容量、MoE dispatch 开销 |
| Apple metal | 跳过 CUDA 向 `modelSizeScaling`；decode 按 MLX 实测比值（约 74–83%）校准 |
| llama.cpp / GGUF | 使用 `gguf_bytes` 计算真实序列化开销；CPU 与分层 Offload 受 DDR 带宽限制 |
| MoE | 按 batch 估算被触达的不同 experts；EP 计入 non-expert 权重复制；CPU Offload 取 DDR 与各卡 PCIe 聚合带宽的较慢者 |
| KV / 状态流量 | Decode 计入序列 KV 读取、新 token KV 写入及循环状态读写；Prefill 在滑动窗口已满后仍计入覆盖写入 |
| MLA / 混合注意力 | 使用显式 latent cache 维度；线性注意力/SSM 状态按固定 FP32 状态计算 |
| 高 batch | `getBatchSchedulingEfficiency` 抑制聚合吞吐虚高 |
| INT4 prefill | 走 BF16 算力而非 INT4 Tensor Core 峰值 |
| 多卡 | 分离物理 GPU 总数、PP、EP、TP，按每卡内存和通信/流水线气泡计算；GQA/MQA 超过 KV head 数时计入复制，MLA 在普通 TP 下按复制 cache 建模 |

运行完整回归（计算不变量、368 个模型、250 个硬件条目、Solver、命令生成和 benchmark）：

```bash
npm test
```

公开 benchmark 数据本身可能来自不同模型修订、量化封装或运行版本；若一个参考值超过所选硬件的原始带宽/算力上限，回归脚本会拒绝把它当作校准目标。

---

## 模型支持范围

**Dense 模型**: Llama 1/2/3/3.1/3.2/3.3、Gemma 1/2/3/4、Qwen 2/2.5/3、Mistral、DeepSeek LLM/Coder/Math 系列、GLM、Baichuan、Bloom、CodeLlama、Falcon、Phi、Yi 等

**MoE 模型**: DeepSeek V2/V3/R1 系列（含 MLA 压缩）、Mixtral、GLM-4/4.5 MoE、Qwen MoE、DBRX、Command R+ 等

**VLM 模型**: LLaVA、LLaMA 3.2 Vision、DeepSeek VL、GLM-4V、CogVLM2 等（含 `vision_seq_tokens` 时按图像数增加上下文与注意力工作量；视觉编码器计算仍标记为近似）

**混合注意力**: Gemma 3（sliding window + global 分层），Qwen3.5/3.6（线性注意力层 linear_attention_layers，FA boost 按比例缩减）

---

## GPU 支持范围

| 厂商 | 系列 |
|------|------|
| NVIDIA | RTX 20/30/40/50 系、Tesla V100、A100、H100/H200、B200/B300 SXM、HGX B300 (8×)、GB200/GB300 NVL72 |
| AMD | RX 5000/6000/7000/9000、MI50/60/100/200/300 系 |
| Intel | Arc A/B 系、Gaudi 2/3 |
| Apple | M1/M2/M3/M4/M5（含 Pro/Max/Ultra 各配置） |
| 国产 | 华为昇腾 910B/C/D/E、壁仞 BR100、寒武纪 MLU370、摩尔线程 S4000 |

GPU 字段支持 `bwUtilization`（实际带宽利用率）、`usableRatio`（可用显存比例）、`decodeBwScale`（同带宽不同 GPU 核数/代际的有效 decode 系数）、`gpuCores`（GPU 核数，用于检测匹配）。

**默认配置**：RTX 4090 × 1 · BF16 · 16K ctx · Gemma 4 12B Unified（`src/pages/Estimator.vue`）。

**本机检测**：WebGPU/WebGL 检测 Apple Silicon 时结合带宽测量 + 内存探测，同内存容量下按带宽区分 Pro 16核/20核等 SKU（`src/utils/detectGpu.js`）。

---

## 扫描功能

- **Batch Sweep**：固定配置下扫描 batch 1→256，输出吞吐/TPOT 曲线 + 数据表，标注 OOM 点和当前 batch
- **GPU 数量扫描**：枚举 1/2/4/8/16/32/64 张卡，输出每档显存和速度

---

## 告警

| 条件 | 级别 |
|------|------|
| 显存不足 | 错误 |
| 显存利用率 > 95% | 警告 |
| 系统 RAM 不足或未指定 | 错误 |
| 上下文、并行拓扑、框架或 KV 格式不兼容 | 错误 |
| 激活内存 > 2 GB | 提示 |
| TP 通信效率 < 70% | 警告 |
| 单请求速度 < 20 tok/s | 警告 |
| Decode Roofline 比 < 0.1（严重带宽瓶颈） | 提示 |
| 总功耗 > 10 kW | 提示 |

---

## 核心文件

| 文件 | 职责 |
|------|------|
| src/utils/calc.js | 核心计算（`calcAll` / `calcBatchSweep` / 混合 GPU 聚合） |
| src/utils/solver.js | 反向求解（显存剪枝对齐 getQuantBytes） |
| scripts/calculation-regression.mjs | 计算公式、物理上限、模型/GPU 全目录回归 |
| scripts/solver-regression.mjs | Solver 拓扑、精度下限和 Pareto 回归 |
| scripts/benchmark-regression.mjs | 公开 benchmark 回归测试 |
| src/utils/model.js | 模型 Attention 类型推断 |
| src/utils/exportMd.js | Markdown 报告导出 |
| src/data/constants.js | 量化精度 / 框架效率系数 / 互联方式常量 |
| src/data/runtime.js | KV Cache 支持矩阵 / PCIe / DDR 代际、速率和 RAM 容量 |
| src/data/gpus/ | GPU 数据（按厂商分目录） |
| src/data/models/ | 模型数据（按模型系列分目录） |

---

## 声明

本工具仅用于容量规划和工程估算；部署前请使用目标模型、运行时与硬件进行实测验证。
