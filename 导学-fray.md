# Fray 项目导学

> 项目性质：纯个人开发的高性能 GPU 算子库（用户提供）  
> 目标岗位：AI Infra 工程师  
> 分析基线：仓库 `main` 分支，HEAD `35c50a7`，工作区另有未提交的 GELU 近似模式改动。本文不把 `third-party/` 中的外部代码计作个人实现，也不把本机未运行的 GPU 结果写成已验证事实。

## 1. 前置知识（面试高频标注）

| 知识点 | 为何需要 | 在本项目中的位置 | 高频度 |
| --- | --- | --- | --- |
| GPU 执行模型：grid、block、warp、线程束分歧 | 解释 tile 如何映射到硬件，以及 `num_warps`、持久化 wave 为什么影响吞吐 | `fray/triton/*.py`、`fray/include/**/*.cuh` | ★★★★★ |
| GPU 存储层次与访存合并 | 算子性能往往由全局内存、共享内存、寄存器和 L2 的数据搬运决定 | GEMM、RoPE/KV-cache、归一化、激活融合实现 | ★★★★★ |
| GEMM 分块与 Tensor Core | 理解 M/N/K tile、K 维循环、FP32 累加和 grouped GEMM | `fray/triton/matmul.py`、`fray/triton/grouped_gemm.py`、`fray/include/gemm/` | ★★★★★ |
| Triton 编程与自动调优 | 这是快速实现和验证算子的主路径之一 | `fray/triton/` | ★★★★★ |
| CUDA/CuTe、NVCC 与动态链接 | 理解 Python 如何生成 CUDA、编译共享库并调用 kernel | `fray/jit/`、`fray/jit_kernels/`、`fray/include/` | ★★★★★ |
| LLM 推理算子链 | 将算子与真实推理阶段连接起来，而不是只讲孤立 kernel | Flash Decoding、Flash MLA、RoPE、RMSNorm、Fused MoE | ★★★★★ |
| MoE 路由与稀疏执行 | 解释 Top-k、dispatch、专家分组计算和加权 combine | `fray/triton/fused_moe.py` | ★★★★★ |
| 数值精度与参考实现 | FP16/BF16、FP32 累加、容差、近似函数决定正确性边界 | `tests/cuda/`、`tests/triton/` | ★★★★☆ |
| Benchmark 方法学 | 区分 kernel-only、prepared、end-to-end，避免混合后端和首轮编译污染 | `fray/utils.py`、各 `test_*_performance`、MoE breakdown | ★★★★★ |
| Python 包构建与 ABI 边界 | 算子库不仅有 kernel，还要处理安装、头文件、参数序列化和版本 | `pyproject.toml`、`setup.py`、`fray/jit/template.py` | ★★★☆☆ |

## 2. 重点亮点与学习顺序（先看这个）

| 亮点标题 | 为什么重要 | 通用技术关键词 | 先看哪些文件 | 建议学习顺序 |
| --- | --- | --- | --- | --- |
| 端到端稀疏执行链路 | 最接近 AI Infra 的完整算子子系统，覆盖路由、调度元数据、两次专家计算与结果聚合 | Top-k routing、prefix sum、grouped GEMM、kernel fusion、host sync | `README.md`、`fray/triton/fused_moe.py`、`tests/triton/test_fused_moe.py` | 1 |
| 即时编译与内容寻址缓存 | 展示从 Python API 到 NVCC、共享库、ABI 调用的系统能力，而非只有 kernel 编写 | codegen、content hash、disk cache、ctypes、ABI | `fray/jit/template.py`、`fray/jit/compiler.py`、`fray/jit/runtime.py`、`tests/cuda/test_jit.py` | 2 |
| 面向形状的参数搜索 | 同一算子在不同形状和 GPU 上需要不同 tile；项目同时包含自研 JIT 调优和 Triton 调优 | autotune、search space、L2 flush、shape specialization | `fray/jit_kernels/tuner.py`、`fray/jit_kernels/fp16_gemm.py`、`fray/triton/matmul.py` | 3 |
| 推理态数据搬运融合 | RoPE 与线性/分页 KV-cache 写回、残差加法与 RMSNorm 融合，能直接解释 launch 和中间张量开销 | paged cache、in-place、fusion、memory bandwidth | `fray/triton/rope.py`、`fray/triton/rmsnorm.py`、对应测试 | 4 |
| 双后端算子验证体系 | CUDA/CuTe 适合底层控制，Triton 适合快速迭代；同一仓库分开实现和测试 | backend separation、reference oracle、accuracy/perf split | `fray/jit_kernels/`、`fray/include/`、`fray/triton/`、`tests/` | 5 |
| 性能证据治理 | 项目已经区分 prepared 与端到端路径，但还缺硬件矩阵、结果归档和 CI，是从作品到工程资产的关键差距 | benchmark fairness、reproducibility、regression gate | `fray/utils.py`、`tests/triton/test_fused_moe.py`、`pyproject.toml` | 6 |

## 3. 必备知识点

- [ ] 能画出 Python 调用、代码生成、NVCC 编译、磁盘缓存、动态加载、CUDA stream 发射的完整链路。
- [ ] 能解释内容哈希中包含哪些因素，哪些变化会触发重新编译。
- [ ] 能解释矩阵乘分块、访存掩码、FP32 累加、warp/stage 配置。
- [ ] 能解释在线 softmax 为什么只需一遍流式统计最大值与归一化和。
- [ ] 能解释 Flash Attention/Decoding 如何避免显式物化完整注意力矩阵。
- [ ] 能从 router logits 一直讲到 MoE 最终 token 输出。
- [ ] 能区分 token 路由元数据、expert offsets 与 GEMM tile 元数据。
- [ ] 能解释为何 host `.item()` 会造成 GPU/CPU 同步，以及无同步 grid 的取舍。
- [ ] 能解释 RoPE 的 NeoX/interleaved 两种布局、partial rotary 与尾部复制。
- [ ] 能说明融合 add + RMSNorm、GELU/SiLU + multiply 的收益来源和副作用边界。
- [ ] 能说明正确性基线、误差容限、非整 tile、非连续输入、原地写回和可变 buffer 重置。
- [ ] 能设计 kernel-only、prepared、end-to-end 三层 benchmark，并保证参考端不泄漏被测后端。

## 4. 推荐阅读（结合仓库）

| 主题 | 通用技术点 | 建议阅读位置 | 预计时间 | 读完能回答什么 |
| --- | --- | --- | --- | --- |
| 项目总览 | 后端划分、算子清单、开发约定 | `README.md` | 20 分钟 | Fray 解决什么问题，边界在哪里？ |
| 公共 API | 模块导出、用户调用面 | `fray/__init__.py`、`fray/triton/__init__.py`、`fray/jit_kernels/__init__.py` | 20 分钟 | 用户从哪里调用，两类后端如何隔离？ |
| JIT 代码生成 | Python 类型到 C ABI、CUDA launcher 生成 | `fray/jit/template.py` | 40 分钟 | Tensor 和 stream 如何跨 Python/CUDA 边界？ |
| JIT 编译缓存 | 编译器探测、架构选择、内容哈希、原子写入 | `fray/jit/compiler.py` | 60 分钟 | 首次调用和缓存命中分别发生什么？ |
| 动态运行时 | `ctypes` 加载、参数校验、进程内缓存 | `fray/jit/runtime.py` | 35 分钟 | 共享库如何加载，ABI 错误如何被拦截？ |
| 自研自动调优 | 搜索空间、并行编译、合法性检查、计时选择 | `fray/jit_kernels/tuner.py` | 60 分钟 | 为什么调优不能只看一次耗时，当前实现还有什么偏差？ |
| CUDA/CuTe 示例链 | wrapper、模板、头文件 kernel 的对应关系 | `fray/jit_kernels/fp16_gemm.py`、`fray/include/gemm/fp16_gemm.cuh`、`tests/cuda/test_fp16_gemm.py` | 75 分钟 | 一个 CUDA/CuTe 算子如何从 Python 一直执行到 GPU？ |
| Triton GEMM | 分块、program id 重排、bias 融合、autotune | `fray/triton/matmul.py`、`tests/triton/test_matmul.py` | 60 分钟 | 非整 tile、stride 输入和 bias 如何处理？ |
| Grouped GEMM | 不均匀专家负载、tile 元数据、主机同步 | `fray/triton/grouped_gemm.py`、`tests/triton/test_grouped_gemm.py` | 70 分钟 | 稀疏专家计算为何需要分组调度？ |
| Fused MoE | 路由、计数、前缀和、dispatch、双 GEMM、combine | `fray/triton/fused_moe.py`、`tests/triton/test_fused_moe.py` | 150 分钟 | 从 logits 到最终输出的所有张量如何流动？ |
| MoE 诊断 | 守护区、诊断元数据、配置扫描 | `tests/triton/test_fused_moe_diagnostics.py` | 60 分钟 | 如何定位越界、空专家、tile 解码或配置问题？ |
| RoPE + KV-cache | 原地/非原地、部分旋转、线性与分页地址映射 | `fray/triton/rope.py`、`tests/triton/test_rope.py` | 120 分钟 | 为什么融合位置编码和 cache 写回，分页地址怎么算？ |
| 归一化与激活融合 | reduction、残差更新、GELU/SiLU 近似 | `fray/triton/rmsnorm.py`、`fray/triton/gelu_mul.py`、`fray/triton/silu_mul.py` | 60 分钟 | 带副作用的融合算子如何保证语义一致？ |
| Attention 家族 | online reduction、decode 分块、分页 MLA | `fray/jit_kernels/flash_decoding.py`、`fray/jit_kernels/flash_mla.py`、`fray/include/flash_attn/`、`fray/include/flash_mla/` | 120 分钟 | prefill 与 decode 的形状和优化重点为何不同？ |
| 测试与计时 | CUDA Event、warmup、L2 flush、参考实现 | `fray/utils.py`、`tests/cuda/`、`tests/triton/` | 90 分钟 | 当前结果哪些可比较，哪些还不能用于简历数字？ |
| 打包与依赖 | 动态版本、头文件打包、第三方 include | `pyproject.toml`、`setup.py`、`uv.lock` | 45 分钟 | 如何安装和分发，当前发布材料缺什么？ |

## 5. 自学提醒

若某文件或原理看不懂，请继续追问 AI；本技能负责给学习路径与题目，不提供逐行讲解。

建议每读完一个主题，强制输出三样东西：一张输入到输出的数据流图、一张 kernel launch/同步点清单、一个能推翻当前性能结论的反例。这样比记函数名更接近 AI Infra 面试要求。

## 6. 项目技术定位

**AI Infra / GPU Kernel 交叉项目。** 依据是项目既实现 CUDA/CuTe 与 Triton 算子，又包含 JIT 编译、缓存、动态加载、自动调优、推理链路融合和端到端 benchmark；它不是模型训练应用，也还不是具备服务化、发布和生产观测的完整推理平台。

### 系统架构

```text
调用方（PyTorch CUDA Tensor）
├── CUDA/CuTe JIT 路径
│   └── Python wrapper
│       -> 模板参数特化与 launcher 代码生成
│       -> 内容哈希 / 内存与磁盘缓存查询
│       -> NVCC 编译 kernel.so
│       -> ctypes ABI 参数校验与动态加载
│       -> 当前 CUDA stream 发射 kernel
│
└── Triton 路径
    └── Python wrapper
        -> shape / dtype / layout 契约检查
        -> grid 与 tile 配置选择或自动调优
        -> Triton JIT 编译与缓存
        -> kernel 发射
        -> 输出张量、原地残差或 KV-cache 写回

横切能力：PyTorch 参考实现、正确性测试、边界用例、CUDA Event benchmark
外部依赖：CUTLASS / CuTe、FlashInfer、ThunderKittens、XQA（位于 third-party，不计入个人实现）
```

### 三条核心执行链路

1. **CUDA/CuTe JIT 链路**：Python wrapper 根据 shape 形成特化 key 与搜索空间，模板层生成带 C ABI 的 CUDA launcher；编译层把源码、Fray 版本、NVCC 路径/版本、编译参数、目标架构等组成内容签名，命中缓存则复用共享库，否则调用 NVCC；运行层通过 `ctypes` 加载并将 Tensor 转为设备指针，在当前 CUDA stream 上执行。
2. **Triton 单算子链路**：公共 wrapper 校验 device、dtype、shape、contiguous/stride 与 alias 约束，根据维度形成 grid；Triton 编译出特化 kernel，kernel 执行 masked load、FP32 中间计算与 masked store；调用方持有预分配输出，部分 API 支持原地更新或 cache 写回。
3. **Fused MoE 端到端链路**：`router_logits [M,E]` 经每 token Top-k 与选中权重归一化，生成专家计数；计数经前缀和得到每个专家的 route 区间，再把 token id/weight scatter 成专家连续的 dispatch 元数据；第一个专家 GEMM 读取原 token、计算 gate/up 并融合 SiLU，第二个专家 GEMM 将结果乘权重并按原 token 累加，得到 `output [M,H]`。无同步 tile 元数据路径用保守 launch 上界和设备侧早退减少 host 同步。

## 7. 核心原理解析

### 7.1 内容寻址的 CUDA JIT

**问题**：不同 GPU 架构、编译器、模板和 tile 配置需要不同二进制，预编译所有组合会膨胀，重复编译又会拖慢启动。  
**机制**：将 kernel 名称、项目源码版本哈希、生成代码、NVCC 版本、编译 flags、平台和优化开关组成签名，再映射到磁盘目录；共享库和参数描述落盘，进程内还保存 Runtime 对象。  
**项目落点**：`fray/jit/compiler.py` 负责签名、编译和原子替换，`fray/jit/runtime.py` 负责有效性检查、加载与调用，`fray/jit/template.py` 负责稳定的参数序列化和 launcher 生成。

### 7.2 形状特化与自动调优

**问题**：固定 tile 无法同时适配小矩阵、方阵、长 K、不同专家负载和不同 GPU。  
**机制**：将稳定的 shape/key 与候选 block/stage/warp 组合分开；先并行编译候选，再做合法性执行和 CUDA Event 计时，选择最快配置并按签名缓存。  
**项目落点**：CUDA/CuTe 路径由 `fray/jit_kernels/tuner.py` 管理搜索，Triton GEMM 使用框架自带 autotune，MoE 对两段 GEMM 的 persistent wave 组合做运行时选择。

### 7.3 稀疏 MoE 的数据重排与计算

**问题**：每个 token 只访问少量专家，专家负载不均；直接逐 token 执行会产生大量小 GEMM 和 launch。  
**机制**：先把 token-expert 路由按专家分组，只存 token id 和权重而不复制完整输入；以专家 offsets 描述变长分段，再把每段映射为 grouped GEMM tile；最后用权重将各 route 输出聚合回 token。  
**项目落点**：`fray/triton/fused_moe.py` 将 Top-k、计数、前缀和、scatter、GEMM1+激活、GEMM2+combine 串为公共端到端 API，并保留 prepared API 做核心计算测量。

### 7.4 融合推理态数据搬运

**问题**：RoPE 后再单独写 KV-cache、残差相加后再做 RMSNorm、激活后再乘上投影分支，都会多读写中间张量并增加 launch。  
**机制**：把共享同一输入/输出的数据变换放入一个 kernel；对线性和分页 cache 分别计算地址；对原地输出显式检查 alias 与尾部复制语义。  
**项目落点**：`fray/triton/rope.py` 覆盖旋转与 KV 写回，`fray/triton/rmsnorm.py` 覆盖残差更新与归一化，`gelu_mul.py`/`silu_mul.py` 覆盖激活乘法。

### 7.5 在线归约与 Attention

**问题**：注意力分数或长向量归约若完整物化中间矩阵，会放大显存流量和容量压力。  
**机制**：按块维护 running max 与 running sum，通过数值稳定的重标定合并分块；decode 路径还要处理 GQA、长 KV 和中间 workspace。  
**项目落点**：`fray/include/softmax/`、`fray/include/flash_attn/`、`fray/include/flash_mla/` 提供底层实现，`fray/jit_kernels/` 提供 Python 侧特化和执行包装。

### 7.6 正确性先于性能结论

**问题**：GPU kernel 可能在典型整形状上正确，却在尾块、非连续 stride、空专家、原地写回或随机分页映射下出错；错误参考实现也会制造虚假加速。  
**机制**：用独立 PyTorch 参考实现、不同 dtype/shape、非整 tile、in-place、paged mapping 和 guard/diagnostic 用例覆盖语义；benchmark 前先比对输出，并拆分 prepared 与端到端路径。  
**项目落点**：`tests/triton/test_rope.py`、`test_fused_moe.py`、`test_fused_moe_diagnostics.py` 等已有对应场景；但自动化 GPU CI 和持久化结果仍缺失。

## 8. 关键设计决策

| 决策 | 备选 | 当前取舍 | 风险 | 如何验证 |
| --- | --- | --- | --- | --- |
| CUDA/CuTe 与 Triton 双路径 | 只保留一个后端 | 分目录隔离：底层控制与快速原型并存 | API/语义重复，维护成本增加 | 建立跨后端同语义 conformance case，标清支持矩阵 |
| 运行时 JIT 而非预编译 wheel | 为固定 SM 预编译二进制 | 按本机架构编译并缓存 | 首次调用延迟、需要 NVCC、部署复杂 | 统计冷启动/热启动、缓存命中率、并发编译行为 |
| 内容哈希缓存 | 仅按 kernel 名缓存 | 源码、版本、编译器和 flags 参与签名 | 多进程并发写、缓存淘汰和磁盘膨胀未治理 | 并发压力测试、损坏恢复、缓存容量与命中率测试 |
| 并行编译调优候选 | 串行编译 | 线程池发起多个 NVCC 子进程 | CPU/内存峰值高，结果未持久化，进程内 GPU 假设 | 记录编译资源、跨进程/多 GPU 复用与结果稳定性 |
| MoE 仅重排索引不复制输入 | 物化按专家排序的 `x` | GEMM 内按 token id 间接读取 | 间接访问可能降低合并度 | 对比 index-only 与 materialized 两条路径，拆分 metadata/GEMM 时间 |
| 无 host sync 的保守 grid | 从 device 读取精确 tile 数 | 用上界发射，设备侧越界早退 | 空 launch/多余 program 带来浪费 | 按负载倾斜和专家数统计有效 tile 比例与端到端收益 |
| 融合 RoPE 与 cache 写回 | 两个独立 kernel | 一次读取/旋转并写入线性或分页 cache | API 组合增多，alias 与地址边界复杂 | 随机 page table、重复位置、越界 guard、两步参考对照 |
| 融合 add + RMSNorm | 两步 PyTorch 或两 kernel | residual 原地更新，output 单独写出 | 可变输入使 benchmark 易失真，alias 可能破坏语义 | 每次迭代重置 residual；显式 alias 拒绝；比较双输出 |
| CUDA Event 平均计时 | profiler、benchmark harness、分位数 | warmup 后剔除最快侧 10% 前段并取均值 | 统计定义不标准，缺置信区间/功耗/频率控制 | 增加 median、p10/p90、bootstrap CI、锁频和 profiler 佐证 |

## 9. 量化与验证（含待测，建议）

### 当前可核验事实

- 仓库当前有 50 个项目 Python 文件，AST 解析无语法错误；`tests/` 中定义了 43 个 pytest 风格函数。
- 分目录收集成功：`tests/cuda` 收集 5 个、`tests/triton` 收集 38 个，共 43 个。当前环境 `torch 2.11.0+cu128`、`triton 3.6.0`，但 `torch.cuda.is_available()` 为 `False`，因此本次没有执行 GPU 正确性或性能测试。
- 将两个测试目录在同一 pytest 进程收集会因 `test_rmsnorm.py`、`test_rope.py`、`test_softmax.py` 同名模块产生 import mismatch；README 当前推荐分目录运行，生产级 CI 仍建议修复命名/包结构或采用 importlib 模式。
- Git 历史有 15 个提交、没有 tag；两个作者名使用同一邮箱。用户声明为纯个人项目，这与提交邮箱归属一致，但“完全独立原创”仍不能仅凭 Git 作者字段证明。
- 历史提交可定位到 reduce、online softmax、GEMM、FlashAttention、RMSNorm、Flash Decoding、Flash MLA、Triton 基础算子、matmul、grouped GEMM；HEAD `35c50a7` 集中加入/重构 JIT、MoE、RoPE、RMSNorm、GELU、测试、README 和锁文件。
- 工作区现有未提交改动位于 `fray/triton/gelu_mul.py` 与 `tests/triton/test_gelu_mul.py`，内容是增加 `none/tanh` 两种 GELU 模式及对应正确性/性能覆盖；本文未修改这两处。

### 已有但需谨慎使用的性能证据

- 既往用户实机反馈记录了：prepared MoE Triton `762.07 us` 对纯 PyTorch `2887.48 us`；从 logits 开始的完整流 Triton `2016.83 us` 对纯 PyTorch `3622.66 us`；add + RMSNorm Triton `681.70 us` 对 PyTorch `6913.59 us`（约 `10.14x`）。
- 这些数字适合当“待复现实验记录”，不宜直接写成无条件简历结论：当前仓库没有绑定 GPU 型号、驱动/CUDA、功耗/频率、shape、commit、原始日志和重复试验分布的结果文件，本次也无法在 CUDA 上复核。

### 建议补齐的指标

| 优先级 | 待测项 | 建议协议 | 可形成的证据 |
| --- | --- | --- | --- |
| P0 | 正确性矩阵 | 每个公共 API 覆盖 dtype、边界 shape、非整 tile、stride/contiguous、in-place/alias、随机种子；对 PyTorch/官方实现 | JSON/JUnit 报告，按 commit 归档 |
| P0 | 端到端延迟 | 固定真实模型 shape，分别报告 cold、warm、prepared、e2e；至少 100 次，给 median/p90/p99 和置信区间 | `benchmarks/results/<gpu>/<commit>.json` |
| P0 | 公平基线 | Triton 对纯 PyTorch/torch.compile/成熟 kernel；参考端不得复用被测 dispatch；两端语义与 buffer 重置一致 | benchmark protocol 与代码审阅清单 |
| P0 | 硬件矩阵 | 至少覆盖一代数据中心卡和一代消费卡；记录 GPU、SM、驱动、CUDA、PyTorch、Triton、功耗/时钟 | 可复现环境清单与原始日志 |
| P1 | 资源效率 | Nsight Compute 记录 DRAM/L2 吞吐、occupancy、tensor core 利用率、register、shared memory | profiler 报告与瓶颈解释 |
| P1 | JIT 成本 | 冷编译耗时、热缓存命中耗时、二进制大小、缓存命中率、并行编译 CPU/内存峰值 | 启动成本与缓存容量曲线 |
| P1 | MoE 分解 | routing、count/prefix、dispatch、tile metadata、GEMM1、GEMM2/combine 分别计时；按负载倾斜变化 | breakdown 图与优化前后对照 |
| P1 | 鲁棒性 | 多进程同时编译同一 key、缓存文件损坏、多 GPU/异构 GPU、OOM/非法配置恢复 | 压力测试和失败恢复报告 |
| P2 | 模型级收益 | 接入一个真实推理框架和开源模型，测 TTFT、TPOT、tokens/s、显存峰值与结果一致性 | 集成 PR、运行脚本、端到端报告 |

### 缺少的上线与开源材料

- **持续集成**：仓库未见项目自有 `.github/workflows`；至少需要 CPU 静态检查/收集任务和目标 GPU runner 的正确性、性能回归任务。
- **发布治理**：无 tag/release、无 changelog、根目录无许可证文件，README 也明确写着 license 尚未提供；这会阻碍外部采用。
- **稳定 API 与兼容矩阵**：缺公开 API 版本策略、支持的 GPU SM/CUDA/PyTorch/Triton 组合、弃用规则和 ABI/cache 迁移说明。
- **可复现 benchmark 工件**：性能用例嵌在测试中并打印到 stdout，缺独立 CLI、结构化结果、硬件元数据、基线锁定和历史趋势图。
- **生产集成**：未见 vLLM/SGLang/TensorRT-LLM 等真实框架接入、模型级精度对照、长稳运行、OOM/降级与回滚方案。
- **可观测性**：缺 JIT cache hit/miss、compile latency、selected config、kernel failure、fallback 等结构化指标和日志接口。
- **安全与供应链**：第三方代码体量大，缺根项目的 SBOM、第三方许可证汇总、来源版本/commit 说明和漏洞扫描。
- **贡献材料**：缺 CONTRIBUTING、代码所有权边界、设计文档、benchmark 复现指南；对个人项目而言，这些也是证明工程成熟度的材料。

### 推荐的最小上线闭环

1. 先把测试命名冲突、根许可证、CI 和硬件/软件兼容矩阵补齐。
2. 抽出独立 benchmark CLI，强制写入 commit、环境、shape、原始样本与正确性结果。
3. 选择一个真实模型链路，只接入 1-2 个最成熟算子，保留 fallback 并做 A/B。
4. 用模型级 TTFT/TPOT/tokens/s、显存、错误率和 1-8 小时稳定性证明价值。
5. 建立性能回归阈值与 release/tag，才把“高性能”从项目目标升级为可持续结论。
