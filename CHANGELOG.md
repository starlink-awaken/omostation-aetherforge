# Changelog

> 所有显著更改都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

---

## [未发布]

### 新增
- AetherForge active 模式现以显式白名单向 omlxcd 透传 OpenAI function tools、
  `tool_choice`、assistant tool calls 与工具结果消息，并在流式及非流式响应中
  保留 tool-call 数据。

### 变更
- 无

### 修复
- 修复 OpenCode 等编码代理在 active 本地链路中工具定义被静默丢弃的问题。
- 修复流式请求在首 token 前收到 typed HTTP 错误时，因响应体尚未读取而把
  `409`/`504` 等安全状态错误泛化为 `502` 的问题；错误体读取限制为 64 KiB。

---
