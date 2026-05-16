PaddleOCR-NPU 更新日志

更新时间：2026-05-16

本次更新重点：多卡多实例 OCR 服务的动态扩缩容能力。

一、多卡扩缩容能力

本服务默认启用 MultiProcessOCRPool。每个 OCR worker 是独立进程，可以按 NPU 卡和当前请求压力动态扩容、缩容。

默认配置：

- NPU 列表：0,1,2,3
- 常驻实例数：min_instances=4
- 最大实例数：max_instances=24
- 单卡最大实例数：per_card_max=6
- 空闲缩容时间：idle_timeout=600 秒，也就是空闲 10 分钟后开始缩容
- 扩容冷却时间：scale_cooldown=5 秒
- 批量请求等待新 worker 时间：batch_acquire_wait=15 秒
- worker ready 后延迟接单时间：worker_assign_delay_sec=5 秒

典型运行日志：

[pool] stats queue=0 waiting=0 worker_q=0 ready=18 assignable=18 busy=0 idle=18 pending=1 per_device={'0': 5, '1': 5, '2': 4, '3': 4}

字段说明：

- queue：当前总排队数
- waiting：正在等待分配 worker 的请求数
- worker_q：已经派给 worker 但还未执行的任务数
- ready：已经初始化完成的 worker 数
- assignable：已经 ready 且过了接单延迟、可以接任务的 worker 数
- busy：正在推理的 worker 数
- idle：当前空闲 worker 数
- pending：还在初始化中的 worker 数
- per_device：每张 NPU 卡上可接单的 worker 分布

上面的日志表示当前没有请求压力，18 个 worker 已就绪且全部空闲，分布在 4 张 NPU 卡上。由于 idle_timeout=600，持续空闲 10 分钟后，池子会逐步回收多余 worker，直到降到 min_instances 或达到稳定状态。

二、扩容策略

- 当请求排队或所有可接单 worker 都忙时，monitor 会尝试扩容。
- 扩容优先选择已有实例较多但仍有 HBM 空间的卡，尽量填满当前卡，避免过度分散。
- per_card_max 会限制单卡最多实例数，默认每张卡最多 6 个 worker。
- instance_hbm_mb 默认 5500，hbm_safety_margin_mb 默认 4096，用于估算每张卡还能否继续拉起 worker。
- 如果 npu-smi 无法读取某张卡 HBM 信息，该卡会作为低优先级兜底候选，而不是直接让整个扩容失败。

三、缩容策略

- worker 必须 idle、ready、未 retiring，且空闲时长超过 idle_timeout 才能被缩容。
- 默认 idle_timeout=600 秒，避免任务刚结束就频繁销毁和重建 OCR 进程。
- 缩容优先回收同卡多实例中的空闲 worker。
- worker 初始化超时后会被 kill 并从池子移除，避免 stats 长期显示 pending=1。
- 服务进入 shutdown 后，monitor/watchdog 不再继续拉起新 worker。

四、OCR 召回参数

本次也调整了默认 OCR 召回参数，优先保障水印、小字、半透明文本和高分辨率截图中的检出率：

- det_limit_side_len=2560
- det_db_thresh=0.12
- det_db_box_thresh=0.15
- det_db_unclip_ratio=1.8
- drop_score=0.0
- max_text_length=64

这些参数会在启动日志中打印，方便确认线上服务实际生效配置。

五、服务状态接口

/info 接口新增进程池状态字段：

- task_queue_size
- waiting_request_count
- ready_instance_count
- assignable_instance_count
- busy_instance_count
- idle_instance_count

这些字段用于快速判断当前 OCR 服务是否正在排队、是否已经扩容、是否存在长时间 pending worker。

六、服务脚本增强

scripts/ocr_service.sh 支持透传以下配置：

- det_limit_side_len
- det_db_thresh
- det_db_box_thresh
- det_db_unclip_ratio
- drop_score
- max_text_length
- min_instances
- max_instances
- per_card_max
- idle_timeout
- scale_cooldown
- batch_acquire_wait
- worker_assign_delay_sec
- stats_log_interval

stop/restart 会优先按进程组停止服务，避免 multiprocessing worker 残留。如果 stop 后端口仍被占用，会升级执行 purge，并兜底清理 python paddle 相关进程，但会排除 funasr 进程。

七、启动和退出日志

- start_server.py 会输出 pid、ppid、pgid。
- uvicorn 收到退出信号时会记录信号名称和进程信息。
- 收到退出信号时会标记 OCR pool 进入 shutdown，避免关闭过程中继续扩容或拉起 worker。

八、运行截图

以下两张截图记录了多卡扩缩容运行状态和 pool stats 输出，可用于确认服务已按多卡多实例模式运行：

- QQ20260516-223832.png
- QQ20260516-223904.png

截图中的典型状态为 ready=18、assignable=18、busy=0、idle=18，per_device={'0': 5, '1': 5, '2': 4, '3': 4}，说明 OCR worker 已经分布到 4 张 NPU 卡上。请求压力消失后，这些空闲 worker 会在 idle_timeout=600 秒后逐步缩容。
