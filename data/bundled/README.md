# CIC 内置兜底样本

`cic_fallback_sample.parquet`（约 1.7 MB）仅在 **所有在线下载均失败** 时使用。

- 设置 `CIC_DOWNLOAD_MODE=bundled` 可强制使用
- **正式实验报告**请优先使用 `hf_minimal`（Hugging Face 官方结构真实数据，按日分片约 55 MB）
