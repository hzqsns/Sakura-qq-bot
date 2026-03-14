# sakura-gemini 重构设计文档

**日期：** 2026-03-14
**目标：** 将 sakura-gemini 的 LLM 调用从直接调用 `provider.text_chat()` 改为通过 AstrBot 内置 Agent 流水线（`build_main_agent`），使 persona、datetime、web 搜索、工具调用等所有框架特性自动生效，消除重复造轮子。

---

## 背景

sakura-gemini 当前在 `on_at_mention` 中拦截事件并直接调用 `provider.text_chat()`，绕过了 AstrBot 的框架流水线。导致：

- `datetime_system_prompt` 不生效（需手动注入）
- persona（人格）不生效
- web 搜索工具不生效
- 工具调用（cron、agent 等）不生效
- 框架新增特性需手动跟进

---

## 职责划分

| 模块 | 职责 |
|------|------|
| `ContextManager` | 群消息录入、用户 Q&A 历史、格式化群聊背景 |
| `main.py` | 事件路由、噪声过滤、管理员命令、冷却、分段发送 |
| **AstrBot 框架** | persona、datetime、web 搜索、工具调用、安全过滤、对话管理 |

sakura-gemini 的核心唯一价值：**自定义群聊上下文**（记录所有人的消息作为 LLM 背景），这是 AstrBot 原生对话管理不具备的能力。

---

## LLM 调用新流程

```
on_at_mention / _try_proactive_reply
    │
    ├─ 构建 ProviderRequest
    │    ├─ prompt          = 用户问题（或 proactive_prompt）
    │    ├─ contexts        = ContextManager 的用户 Q&A 历史（OpenAI 格式）
    │    ├─ image_urls      = 图片 URL 列表
    │    └─ extra_user_content_parts
    │         └─ TextPart("<group_context>\n近期群聊记录\n</group_context>")
    │
    ├─ _build_agent_cfg()
    │    └─ 从 self.context.get_config() 读取全局 provider_settings 等
    │    └─ 返回 MainAgentBuildConfig
    │
    ├─ build_main_agent(event, self.context, cfg, req)
    │    └─ AstrBot 注入：persona / datetime / tools / prompt_prefix / web搜索
    │    └─ 返回 MainAgentBuildResult(agent_runner, provider_request, provider, reset_coro)
    │
    ├─ await reset_coro（必须在 run_agent 前执行）
    │
    ├─ async for _ in run_agent(agent_runner, show_tool_use=False): pass
    │    └─ 处理工具调用循环（web 搜索等），只 drain，不自己 yield 中间结果
    │
    └─ reply_text = agent_runner.get_final_llm_resp().completion_text
```

---

## 代码结构变更

### 新增导入

```python
from astrbot.core.astr_main_agent import build_main_agent, MainAgentBuildConfig
from astrbot.core.provider.entities import ProviderRequest, TextPart
from astrbot.core.astr_agent_run_util import run_agent
```

### 删除导入

```python
from datetime import datetime, timezone, timedelta  # datetime 交给框架
```

### 新增辅助方法

**`_build_agent_cfg(self) -> MainAgentBuildConfig`**

从 `self.context.get_config()` 读取全局配置，构建 `MainAgentBuildConfig`。
关键字段：`provider_settings`、`tool_call_timeout`、`add_cron_tools`、`timezone` 等。
此方法在 `__init__` 后首次调用时构建并缓存（配置热重载时重新构建）。

**`async def _call_agent(self, event, prompt, contexts, group_context_text, image_urls) -> str | None`**

统一 LLM 调用入口：
1. 构建 `ProviderRequest`
2. 注入 `group_context_text` 到 `extra_user_content_parts`
3. 调用 `build_main_agent`
4. 等待 `reset_coro`
5. Drain `run_agent`
6. 返回 `completion_text`，失败时返回 `None`

### `_handle_query` 变更

- 移除 `beijing_tz` / `dynamic_system_prompt` 手动 datetime 注入
- 移除 `provider = self.context.get_using_provider(...)` 直接获取
- 移除 `await provider.text_chat(...)` 直接调用
- 改为调用 `await self._call_agent(event, question, user_contexts, group_ctx, image_urls)`
- 其余逻辑（冷却、ContextManager 录入、分段发送）保持不变

### `_try_proactive_reply` 变更

- 移除 `provider = self.context.get_using_provider(...)` 直接获取
- 移除手动构建 contexts 列表和 `await provider.text_chat(...)`
- 改为调用 `await self._call_agent(event, self.proactive_prompt, [], group_ctx, [])`
- 其余逻辑（计数器、概率门、冷却）保持不变

### `__init__` 变更

- 移除 `self.system_prompt`（persona 完全交给 AstrBot WebUI 管理）
- `_agent_cfg` 延迟初始化（首次 `_call_agent` 时构建）

### `_conf_schema.json` 变更

- 删除 `system_prompt` 字段
- 其余字段保持不变

### `ContextManager` 变更

- 新增 `format_group_context(group_id: str, n: int) -> str` 方法
  - 取最近 n 条群消息，格式化为 `[sender_name]: content` 行
  - 返回空字符串表示无历史

---

## 配置变更

### 移除

| 字段 | 原因 |
|------|------|
| `system_prompt` | persona 由 AstrBot WebUI 的人格系统管理 |

### 保留

`cooldown_seconds`, `segment_length`, `min_msg_length`, `proactive_every_n`, `proactive_probability`, `render_probability`, `delegate_to_angel_heart`, `admin_qq_list`, `group_ctx_max`, `user_ctx_max`, `ctx_expire_seconds`

---

## 边界情况处理

| 情况 | 处理方式 |
|------|----------|
| `build_main_agent` 返回 `None`（未配置 provider） | 回复「未配置 AI 服务，请联系管理员」 |
| `get_final_llm_resp()` 返回 `None` | 回复「抱歉，我无法生成回复。」 |
| `run_agent` 抛出异常 | catch → logger.error → 回复「请求失败，请稍后再试」 |
| 旧配置中存在 `system_prompt` 字段 | 静默忽略，不读取 |
| `group_context_text` 为空 | 不注入 `<group_context>` 块，正常继续 |

---

## 迁移说明

- 用户需在 AstrBot WebUI → 人格 中配置英梨梨人格（已在之前会话中创建）
- 旧的 `system_prompt` 插件配置项自动失效，无需手动清除
- 所有其他配置项保持兼容

---

## 不在本次范围内

- ContextManager 的 DB schema 变更
- 测试用例更新（contexts 格式变化后需同步 test_context.py）
- angel_heart delegate 模式的变更
