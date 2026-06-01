# Hermes 记忆系统深度分析与优化方案

> 分析日期: 2026-06-01 | 范围: 纯记忆系统 | 方法: 代码审计 + 文献调研 + 逻辑推理

---

## 一、架构全景评估

### 1.1 三层架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    Tier 1: Built-in Memory                   │
│  MEMORY.md / USER.md → flat files, § delimited              │
│  Frozen snapshot at session start, atomic writes             │
│  Always-on, no external dependencies                         │
├─────────────────────────────────────────────────────────────┤
│                    Tier 2: Provider Plugin System             │
│  MemoryProvider ABC → 8 external providers                   │
│  MemoryManager orchestrates lifecycle                        │
│  One external provider at a time                             │
├─────────────────────────────────────────────────────────────┤
│                    Tier 3: Organic Memory Pipeline           │
│  MemoryPipeline (interceptor in MemoryManager)               │
│  9 neuroscience-inspired layers                              │
│  Operates on ALL memory pathways                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 架构优势（值得保留的设计）

| 设计决策 | 评价 |
|---------|------|
| 三层解耦：Built-in / Provider / Pipeline | ✅ 优秀。Built-in 保证最低可用性，Pipeline 是可选增强层 |
| Provider 插件化 + 单一活跃约束 | ✅ 正确。避免多 provider 冲突，降低认知负担 |
| Frozen snapshot 保护 LLM prefix cache | ✅ 关键优化。避免每次 turn 重建 system prompt |
| Pipeline 作为 interceptor 而非 provider | ✅ 正确。Pipeline 增强所有 pathway，不替代任何 provider |
| 神经科学启发的分层设计 | ✅ 理论基础扎实。Salience → Engram → Consolidation → Reconsolidation → Feedback → Activation |

### 1.3 核心架构缺陷

| 缺陷 | 影响 |
|------|------|
| Pipeline 2586 行单文件 | 🔴 可维护性极差。9 个层全部耦合在一个文件中 |
| 全异常被 `logger.debug` 吞噬 | 🔴 生产环境完全不可观测。Pipeline 故障无任何告警 |
| SQLite 连接无健康检查/重连 | 🔴 一旦连接中断，Pipeline 永久失效且无恢复路径 |
| Holographic 插件 4 个模块是死代码 | 🟡 episodic/dreaming/self_evolution/hippocampal_index 从未被 Provider 接入 |
| 跨模块表依赖未声明 | 🟡 dreaming.py 引用 memory_pipeline.py 创建的表，无文档、无校验 |

---

## 二、问题分类学（按根因，非按严重度）

### 2.1 并发安全类（5 个问题，3 Critical + 2 High）

**根因：** SQLite 连接跨线程共享，但锁的获取不一致、不完整。

| ID | 问题 | 位置 | 严重度 |
|----|------|------|--------|
| C1 | PipelineState 多 SQL 操作未在同一锁事务中 | memory_pipeline.py:330-376, 666-726 | Critical |
| C3 | FeedbackCoordinator 锁顺序反转 (state._lock ↔ self._lock) | memory_pipeline.py:1048 vs 1140 | Critical |
| H-RT1 | retrieval.py 7 处直接访问 store._conn 无锁 | retrieval.py:416,470,547,620,695,789,831 | High |
| H-RT2 | Lazy SentenceTransformer 实例化非线程安全 | store.py:615-634 | Medium |
| M9 | Windows msvcrt.locking 失败后静默放弃互斥 | memory_tool.py:229-242 | High |

**科学推理：** SQLite 在 WAL 模式下支持并发读，但写操作仍需串行化。当前设计用 `check_same_thread=False` 绕过了线程检查，但未提供一致的并发控制。C3 的锁顺序反转是教科书级死锁条件——在高并发 subagent 场景下必然触发。

### 2.2 算法正确性类（8 个问题）

| ID | 问题 | 影响 |
|----|------|------|
| H1/L11 | Salience 重复信号双重计数：novelty 内含 freshness，raw 又乘 rep_factor | 重复消息被过度惩罚 |
| H2 | Schema 去重仅用前 50 字符 | 不同事实共享前缀被误判为重复 |
| H3 | 新 engram 强度恒为 1.0（min(1.0, 1.0+delta)） | 新记忆无"脆弱期"，违反 Ebbinghaus 遗忘曲线 |
| H4 | ActivationGraph 最短路径权重方向反 | 找到的是最弱路径而非最强路径 |
| H5 | 预测失败时 ALL schemas 信心降低 | 无关 schema 被无辜惩罚，信心持续通缩 |
| M2 | 情绪衰减用当前消息的情绪值作用于所有记忆 | 历史记忆的衰减被当前会话情绪"污染" |
| D2 | schemas.updated_at 从不更新 | predict() 的时间窗口查询永远只匹配新创建的 schema |
| D4 | predict() 的 schema_id 查询 LIKE 方向反 | schema_id 几乎永远为 None，预测无法回溯到 schema |

**科学推理：** H3 违反了记忆科学的核心原则——新记忆应该是脆弱的（Ebbinghaus 1885），需要通过巩固来强化。当前实现让新记忆直接跳到最大强度，跳过了"脆弱窗口"，这意味着系统无法区分"刚记住"和"已巩固"的记忆。H5 的全 schema 信心降低会导致"习得性无助"效应——所有 schema 的信心持续下降，最终都接近 0.1 下限，使信心评分失去区分度。

### 2.3 资源管理类（4 个问题）

| ID | 问题 | 影响 |
|----|------|------|
| C2 | Pipeline 初始化异常时 SQLite 连接泄漏 | 连接数累积，最终耗尽文件描述符 |
| H6 | shutdown() 不等待后台线程 | dreaming/self-evolution 线程操作已关闭的连接 |
| H-RT4 | Holographic shutdown 不关闭 DB 连接 | WAL 文件锁定，连接泄漏 |
| M6 | PipelineState 无连接重连逻辑 | 一次中断永久失效 |

### 2.4 功能完整性类（6 个问题）

| ID | 问题 | 影响 |
|----|------|------|
| M3 | Activation expansion 计算后从未注入响应 | 计算浪费，功能未闭环 |
| D3 | CrossDomainLinks 写入后从未被有意义地读取 | 写放大，存储浪费 |
| M4 | post_delegation 是空操作 | 子代理结果未被记忆系统捕获 |
| H-RT3 | Holographic 4 个模块是死代码 | 占用代码空间，误导开发者 |
| M7 | 每次 consolidation 限 10 条事实 | 大量积压时处理过慢 |
| M1 | 中文情绪模式缺少"宕机/超时/死锁"等 | 中文技术对话的紧急度评分偏低 |

### 2.5 检索质量类（5 个问题）

| ID | 问题 | 影响 |
|----|------|------|
| M11 | 实体提取仅识别首字母大写英文词 | "docker", "python", "API" 等全大写/全小写词被忽略 |
| H-RT5 | 实体提取纯 regex，无 NER | 大量误提取 |
| H-RT6 | FTS5 查询注入 | 用户输入被解释为布尔查询 |
| H-RT9 | FTS5 默认 tokenizer 不支持 CJK 分词 | 中文搜索质量极差 |
| H-RT7 | Trust 评分无时间衰减 | 过时事实永远保持高 trust |

---

## 三、优化路线图

### 阶段划分原则

1. **Phase 0 (止血):** 修复 Critical/High 级并发和资源泄漏问题
2. **Phase 1 (算法修正):** 修复算法正确性问题，恢复预期行为
3. **Phase 2 (架构重构):** 拆分文件、统一错误处理、完善生命周期
4. **Phase 3 (功能闭环):** 激活死代码、完善中文化、优化检索
5. **Phase 4 (科学增强):** 基于文献调研的高级优化

---

### Phase 0: 止血（预计 2-3 天）

#### 0.1 统一并发控制

**问题：** C1, C3, H-RT1, M9
**方案：** 引入统一的数据库访问层

```python
# memory_pipeline.py — 新增 DatabaseAccessor
class DatabaseAccessor:
    """Thread-safe SQLite accessor with consistent locking."""

    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock):
        self._conn = conn
        self._lock = lock

    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def execute_many(self, sql: str, params_seq) -> None:
        with self._lock:
            self._conn.executemany(sql, params_seq)

    def transaction(self, operations: list[tuple[str, tuple]]) -> None:
        """Execute multiple SQL operations in a single locked transaction."""
        with self._lock:
            try:
                for sql, params in operations:
                    self._conn.execute(sql, params)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
```

**关键修改点：**
- `PipelineState` 暴露 `db: DatabaseAccessor` 而非直接暴露 `_conn`
- `retrieval.py` 所有方法改用 `self.store.db.execute()` 替代 `self.store._conn.execute()`
- 消除 C3 的锁顺序反转：`FeedbackCoordinator` 不再直接持有 `state._lock`，全部通过 `DatabaseAccessor`

#### 0.2 修复资源泄漏

**问题：** C2, H6, H-RT4
**方案：**

```python
# memory_pipeline.py — MemoryPipeline.initialize()
def initialize(self, session_id: str, **kwargs) -> None:
    try:
        self._state = PipelineState(db_path)
        self._init_core_layers()
        self._init_plugin_layers()
        self._session_id = session_id
    except Exception:
        # 关键：初始化失败时立即清理
        if self._state is not None:
            self._state.close()
            self._state = None
        raise  # 向上传播，不要吞掉
```

```python
# memory_pipeline.py — shutdown() 等待后台线程
def shutdown(self) -> None:
    # 等待后台线程完成（最多 5 秒）
    for thread in self._background_threads:
        thread.join(timeout=5.0)
    if self._state is not None:
        self._state.close()
        self._state = None
```

```python
# plugins/memory/holographic/__init__.py — shutdown 关闭连接
def shutdown(self) -> None:
    if self._retriever is not None:
        self._retriever.close_cache()
    if self._store is not None:
        self._store.close()
    self._store = None
    self._retriever = None
```

#### 0.3 修复 msvcrt.locking 静默失败

**问题：** M9
**方案：**

```python
# tools/memory_tool.py — _file_lock
try:
    fd.seek(0)
    msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
    yield
finally:
    try:
        fd.seek(0)
        msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
    except (OSError, IOError):
        pass
except (OSError, IOError) as e:
    # 关键修改：锁获取失败时抛出异常，而非静默继续
    raise RuntimeError(f"Failed to acquire file lock: {e}") from e
```

---

### Phase 1: 算法修正（预计 3-4 天）

#### 1.1 修复 Salience 双重计数

**问题：** H1/L11
**当前代码：**
```python
novelty = freshness  # freshness 来自 rep.observe()
rep_factor = freshness
raw = (0.25 * emotion + 0.30 * novelty + 0.30 * importance
       + 0.15 * min(1.0, len(text) / 200))
adjusted = raw * rep_factor  # freshness 被算了两次
```

**修正方案：** 将 novelty 和 rep_factor 解耦
```python
freshness = self._rep.observe(text)  # [0, 1]，越高越新
recency_boost = ...  # 时间衰减
novelty = min(1.0, freshness + recency_boost)  # 新鲜度 + 时间

# novelty 已包含 freshness 信号，不再额外乘 rep_factor
# 改为：rep_factor 仅在 is_trivial 时作为额外惩罚
rep_penalty = 1.0 if not is_trivial else freshness
adjusted = raw * rep_penalty * trivial_multiplier
```

**理论依据：** 信号不应在特征层和输出层被重复编码。新鲜度要么作为特征（novelty），要么作为后处理乘子（rep_factor），不能同时出现在两处。

#### 1.2 修复新 engram 强度

**问题：** H3
**当前代码：**
```python
new_str = min(1.0, 1.0 + delta)  # 永远 = 1.0
```

**修正方案：**
```python
# 新 engram 从 0.3 开始（脆弱期），通过检索/巩固逐步增强
INITIAL_ENGRAM_STRENGTH = 0.3

def strengthen(self, memory_ref: str, delta: float = 0.03) -> float:
    row = state._conn.execute(
        "SELECT strength FROM engram_strengths WHERE memory_ref = ?",
        (memory_ref,)
    ).fetchone()
    if row:
        new_str = min(1.0, row["strength"] + delta)
        # UPDATE ...
    else:
        new_str = min(1.0, INITIAL_ENGRAM_STRENGTH + delta)
        # INSERT with new_str
```

**理论依据：** Ebbinghaus (1885) 的遗忘曲线表明新记忆最脆弱。FSRS 模型中，新记忆的 Stability 从低值开始，每次成功检索后乘性增长。0.3 的初始值意味着新记忆在默认 half_life 后约 30% 的概率被检索到，符合"刚记住但不牢固"的直觉。

#### 1.3 修复 Schema 去重

**问题：** H2
**当前代码：**
```python
existing_contents = {r["content"][:50] for r in existing}  # 只看前 50 字符
if content[:50] in existing_contents:
    continue  # 误判为重复
```

**修正方案：** 使用 content hash + 语义相似度双重检测
```python
import hashlib

def _content_hash(text: str) -> str:
    """Normalized content hash for exact dedup."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]

# 在 consolidate() 中:
existing_hashes = set()
existing_contents = []
for r in state._conn.execute(
    "SELECT schema_id, content FROM schemas ORDER BY updated_at DESC LIMIT 100"
):
    existing_hashes.add(_content_hash(r["content"]))
    existing_contents.append((r["schema_id"], r["content"]))

for fact in facts_sorted:
    content = fact["content"]
    h = _content_hash(content)
    if h in existing_hashes:
        continue  # 精确重复

    # 语义近似检测（仅对 hash 不匹配的）
    is_approx_dup = False
    for sid, ec in existing_contents:
        if _char_similarity(content, ec) > 0.85:  # 字符级 Jaccard
            is_approx_dup = True
            # 合并而非丢弃：更新已有 schema 的 confidence
            state._conn.execute(
                "UPDATE schemas SET confidence = MIN(1.0, confidence + 0.05), "
                "updated_at = CURRENT_TIMESTAMP WHERE schema_id = ?", (sid,)
            )
            break
    if is_approx_dup:
        continue
```

#### 1.4 修复 ActivationGraph 权重方向

**问题：** H4
**当前代码：**
```python
path = nx.shortest_path(G, source=entity_a, target=entity_b, weight="weight")
# weight = strength，所以找到的是最弱路径
```

**修正方案：**
```python
# 构建图时存储 inverse weight
for row in rows:
    G.add_edge(row["entity_a"], row["entity_b"],
               weight=1.0 / max(row["strength"], 0.01),  # 避免除零
               strength=row["strength"])

# shortest_path 现在正确找到最强路径（因为 1/强 = 小权重 = 短距离）
path = nx.shortest_path(G, source=entity_a, target=entity_b, weight="weight")
```

#### 1.5 修复 Schema 信心通缩

**问题：** H5
**当前代码：**
```python
# 预测失败时，降低所有 schema 的信心
state._conn.execute(
    "UPDATE schemas SET confidence = MAX(0.1, confidence - 0.05) "
    "WHERE confidence > 0.3"
)
```

**修正方案：** 只降低产生预测的 schema 的信心
```python
# 从 predictions 表获取关联的 schema_id
prediction = state._conn.execute(
    "SELECT schema_id FROM predictions WHERE prediction_id = ?",
    (prediction_id,)
).fetchone()

if prediction and prediction["schema_id"]:
    state._conn.execute(
        "UPDATE schemas SET confidence = MAX(0.1, confidence - 0.05), "
        "updated_at = CURRENT_TIMESTAMP WHERE schema_id = ?",
        (prediction["schema_id"],)
    )
else:
    # 无法追溯到具体 schema 时，降低信心最高的 3 个 schema
    state._conn.execute(
        "UPDATE schemas SET confidence = MAX(0.1, confidence - 0.02) "
        "WHERE schema_id IN (SELECT schema_id FROM schemas "
        "ORDER BY confidence DESC LIMIT 3)"
    )
```

#### 1.6 修复 Schema updated_at

**问题：** D2
**方案：** 在所有 UPDATE schemas 语句中添加 `updated_at = CURRENT_TIMESTAMP`

```python
# 全局搜索所有 "UPDATE schemas SET" 语句，确保都包含:
# updated_at = CURRENT_TIMESTAMP
```

#### 1.7 修复 predict() 的 schema_id 查询

**问题：** D4
**当前代码：**
```python
"SELECT schema_id FROM schemas "
"WHERE ? LIKE '%' || substr(content, 1, 50) || '%' "
```
LIKE 方向反了——检查的是 pred_text 是否包含 schema 前缀。

**修正方案：**
```python
# 正向匹配：schema 内容是否包含在预测文本中
"SELECT schema_id FROM schemas "
"WHERE substr(content, 1, 50) LIKE '%' || ? || '%' "
"ORDER BY confidence DESC LIMIT 1",
(pred_text[:50],)  # 用预测文本的前 50 字符匹配
```

#### 1.8 修复情绪衰减的全局污染

**问题：** M2
**当前代码：** `apply_decay(emotional_valence=result.emotion)` 对所有记忆使用当前消息的情绪。

**修正方案：** 使用 per-memory 情绪值
```python
# 在 pre_sync() 中改用 apply_decay_with_emotion()
# 需要在 engram_strengths 表中存储每条记忆的情绪值
self._engram.apply_decay_with_emotion(state)  # 读取每条记忆自身的情绪
```

需要修改 `engram_strengths` 表，增加 `emotional_valence` 列：
```sql
ALTER TABLE engram_strengths ADD COLUMN emotional_valence REAL DEFAULT 0.0;
```

---

### Phase 2: 架构重构（预计 5-7 天）

#### 2.1 拆分 memory_pipeline.py

**当前状态：** 2586 行单文件，9 个类
**目标结构：**

```
agent/
├── memory_pipeline.py          # MemoryPipeline 主类 (~200 行)
├── pipeline/
│   ├── __init__.py
│   ├── state.py                # PipelineState, DatabaseAccessor (~200 行)
│   ├── salience.py             # SalienceScorer, _RepetitionDetector (~200 行)
│   ├── engram.py               # SilentEngramEngine (~200 行)
│   ├── consolidation.py        # ConsolidationEngine, DeepConsolidationEngine (~300 行)
│   ├── reconsolidation.py      # ReconsolidationEngine (~200 行)
│   ├── feedback.py             # FeedbackCoordinator (~250 行)
│   ├── activation.py           # ActivationGraph (~250 行)
│   └── sleep.py                # SleepScheduler (~150 行)
```

#### 2.2 统一错误处理

**当前问题：** 所有异常被 `logger.debug` 吞噬
**方案：** 分级错误处理策略

```python
# pipeline/state.py
class PipelineErrorHandler:
    """分级错误处理：区分可恢复和不可恢复错误。"""

    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._consecutive_failures = 0
        self._max_failures_before_disable = 10

    def handle(self, operation: str, error: Exception, context: dict = None):
        self._consecutive_failures += 1

        if isinstance(error, (sqlite3.OperationalError, sqlite3.DatabaseError)):
            # 数据库错误：WARNING 级别，尝试重连
            self._logger.warning("Pipeline DB error in %s: %s", operation, error)
            self._attempt_reconnect()
        elif isinstance(error, (MemoryError, OSError)):
            # 资源错误：ERROR 级别
            self._logger.error("Pipeline resource error in %s: %s", operation, error)
        else:
            # 逻辑错误：WARNING 级别
            self._logger.warning("Pipeline logic error in %s: %s", operation, error)

        if self._consecutive_failures >= self._max_failures_before_disable:
            self._logger.error("Pipeline disabled after %d consecutive failures",
                             self._consecutive_failures)
            raise PipelineDisabledError("Too many consecutive failures")

    def success(self):
        self._consecutive_failures = 0
```

#### 2.3 完善生命周期管理

**方案：** 为所有组件引入统一的 Lifecycle 协议

```python
# pipeline/base.py
from abc import ABC, abstractmethod
from typing import Optional

class PipelineComponent(ABC):
    """统一的 Pipeline 组件生命周期协议。"""

    def initialize(self, state: 'PipelineState', session_id: str) -> None:
        """初始化组件，接收共享的 PipelineState。"""
        self._state = state
        self._session_id = session_id

    @abstractmethod
    def process(self, *args, **kwargs):
        """核心处理逻辑。"""
        ...

    def shutdown(self) -> None:
        """清理资源。子类可覆盖。"""
        pass

    def health_check(self) -> bool:
        """健康检查。返回 True 表示组件正常。"""
        return self._state is not None
```

**修改所有层类继承 PipelineComponent，消除 L7/L8 的私有属性直接赋值问题。**

---

### Phase 3: 功能闭环（预计 3-4 天）

#### 3.1 激活 Holographic 死代码

**当前问题：** episodic.py, dreaming.py, self_evolution.py, hippocampal_index.py 从未被 Provider 接入。

**方案：** 在 HolographicMemoryProvider 中可选启用

```python
# plugins/memory/holographic/__init__.py
class HolographicMemoryProvider(MemoryProvider):
    def initialize(self, session_id: str, **kwargs) -> None:
        # ... 现有初始化 ...

        # 可选启用高级功能
        if self._config.get("episodic_enabled", False):
            from .episodic import EpisodicTimeline
            self._episodic = EpisodicTimeline(self._store._conn, self._store._lock)

        if self._config.get("dreaming_enabled", False):
            from .dreaming import DreamEngine
            self._dreaming = DreamEngine(self._store._conn, self._store._lock)
```

#### 3.2 修复 CJK 搜索

**问题：** FTS5 默认 tokenizer 不支持中文分词
**方案：** 使用 `unicode61` 的 `tokenchars` 选项或引入 jieba 分词

```python
# store.py — 建表时配置 tokenizer
"""
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    content, category,
    content=facts, content_rowid=fact_id,
    tokenize='unicode61 tokenchars 。，！？、；：""''（）【】《》'  -- 包含中文标点
)
"""
# 更优方案：应用层分词后存入 FTS5
def _tokenize_for_fts(self, text: str) -> str:
    """中文分词后用空格连接，供 FTS5 索引。"""
    try:
        import jieba
        return " ".join(jieba.cut(text))
    except ImportError:
        # 回退：按字符分隔
        return " ".join(text)
```

#### 3.3 FTS5 查询消毒

**问题：** H-RT6
**方案：**

```python
def _sanitize_fts5_query(query: str) -> str:
    """Escape FTS5 special characters in user queries."""
    # 转义 FTS5 特殊字符
    special_chars = ['"', "'", '(', ')', '*', '+', '-', ':', '^', '{', '}', '~']
    result = query
    for ch in special_chars:
        result = result.replace(ch, f'"{ch}"')
    return result
```

#### 3.4 增强中文情绪模式

**问题：** M1
**方案：**

```python
_EMOTION_PATTERNS_ZH = [
    # 紧急/严重 (severity=3)
    (re.compile(r'宕机|崩溃|死锁|超时|断线|数据丢失|安全事故|生产事故', re.I), 0.95),
    # 强负面 (severity=2)
    (re.compile(r'严重|紧急|危险|失败|出错|报错|异常|故障|漏洞', re.I), 0.85),
    # 负面 (severity=1)
    (re.compile(r'问题|困难|担心|疑虑|不确定|卡住|慢', re.I), 0.70),
    # 正面
    (re.compile(r'成功|完成|解决|修复|优化|提升|很好|优秀', re.I), 0.30),
]
```

#### 3.5 为 Trust 评分添加时间衰减

**问题：** H-RT7
**方案：**

```python
def get_effective_trust(self, fact_id: int) -> float:
    """获取考虑时间衰减的有效 trust 评分。"""
    row = self._conn.execute(
        "SELECT trust, updated_at FROM facts WHERE fact_id = ?", (fact_id,)
    ).fetchone()
    if not row:
        return 0.0

    base_trust = row["trust"]
    updated_at = datetime.fromisoformat(row["updated_at"])
    age_days = (datetime.now(timezone.utc) - updated_at).total_seconds() / 86400

    # 30 天 half-life 的时间衰减
    decay_factor = math.pow(2, -age_days / 30.0)
    return base_trust * decay_factor
```

---

### Phase 4: 科学增强（预计 5-7 天）

#### 4.1 引入 FSRS 风格的记忆强度模型

**理论基础：** Free Spaced Repetition Scheduler (Jarrett Ye, 2022-2023) 的 DSR 模型比当前的简单 power-law 更精确。

**当前模型：** 单参数 decay（half_life_hours），所有记忆共享同一衰减曲线。
**FSRS 模型：** 三参数 —— Difficulty（固有难度）、Stability（当前稳定性）、Retrievability（当前可检索性）。

```python
# pipeline/engram.py — FSRS 增强
@dataclass
class MemoryStrength:
    difficulty: float    # [0, 1] —— 记忆的固有难度，高 = 难记
    stability: float     # [0, ∞) —— 当前稳定性（天），高 = 衰减慢
    retrievability: float # [0, 1] —— 当前可检索概率

    def decay(self, elapsed_days: float) -> float:
        """计算经过 elapsed_days 天后的可检索性。"""
        if self.stability <= 0:
            return 0.0
        return math.pow(1 + elapsed_days / (9 * self.stability), -1)

    def review(self, rating: int) -> 'MemoryStrength':
        """检索后更新强度。rating: 1=忘记, 2=困难, 3=良好, 4=容易。"""
        new_d = self._update_difficulty(rating)
        new_s = self._update_stability(rating)
        return MemoryStrength(difficulty=new_d, stability=new_s, retrievability=1.0)
```

**迁移策略：** 现有 `engram_strengths` 表增加 `difficulty` 和 `stability` 列，旧数据通过 `stability = half_life / ln(2)` 转换。

#### 4.2 增强 Activation Graph 为 PMI 加权

**理论基础：** Collins & Loftus (1975) 的 spreading activation + PMI（Pointwise Mutual Information）加权比简单共现计数更能反映实体间的语义关联强度。

```python
# pipeline/activation.py
def update_coactivation(self, entity_a: str, entity_b: str) -> None:
    """使用 PMI 加权更新共激活边。"""
    # 获取边缘出现频率
    count_a = self._get_entity_frequency(entity_a)
    count_b = self._get_entity_frequency(entity_b)
    count_ab = self._get_cooccurrence_frequency(entity_a, entity_b)
    total = self._get_total_cooccurrences()

    if count_a == 0 or count_b == 0 or count_ab == 0:
        return

    # PMI = log(P(a,b) / (P(a) * P(b)))
    pmi = math.log2((count_ab / total) / ((count_a / total) * (count_b / total)))
    pmi = max(0, pmi)  # 只保留正相关

    # 边权重 = PMI 的 sigmoid 映射到 [0, 1]
    weight = 1.0 / (1.0 + math.exp(-pmi))

    self._state.execute(
        "INSERT INTO activation_edges (entity_a, entity_b, strength, last_activated) "
        "VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(entity_a, entity_b) DO UPDATE SET "
        "strength = ?, last_activated = CURRENT_TIMESTAMP",
        (entity_a, entity_b, weight, weight)
    )
```

#### 4.3 引入 NLI 矛盾检测

**理论基础：** 当前的 ReconsolidationEngine 使用 embedding similarity + LLM 判断。研究表明 NLI（Natural Language Inference）分类器在矛盾检测上更精确（SNLI/MNLI ~90% 准确率）。

```python
# pipeline/reconsolidation.py
class NLIDetector:
    """轻量级 NLI 矛盾检测。"""

    def __init__(self):
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder('cross-encoder/nli-deberta-v3-small')

    def check_contradiction(self, text_a: str, text_b: str) -> dict:
        """返回 {'contradiction': float, 'entailment': float, 'neutral': float}"""
        self._ensure_model()
        scores = self._model.predict([(text_a, text_b)])
        # scores 是 [contradiction, entailment, neutral] 的 logits
        return {
            'contradiction': float(scores[0][0]),
            'entailment': float(scores[0][1]),
            'neutral': float(scores[0][2]),
        }
```

#### 4.4 实体提取增强

**当前问题：** 纯 regex，仅识别首字母大写英文词
**方案：** 分层提取策略

```python
# pipeline/entity_extraction.py
class EntityExtractor:
    """分层实体提取：规则 → NER → LLM fallback。"""

    def __init__(self):
        self._ner_model = None

    def extract(self, text: str) -> list[dict]:
        entities = []

        # Layer 1: 规则提取（快速，无模型依赖）
        entities.extend(self._rule_based_extract(text))

        # Layer 2: NER（精确，需要模型）
        entities.extend(self._ner_extract(text))

        # 去重 + 合并
        return self._deduplicate(entities)

    def _rule_based_extract(self, text: str) -> list[dict]:
        """改进的规则提取，支持全大写缩写和技术术语。"""
        patterns = [
            # 全大写缩写：API, SQL, HTTP, JSON
            re.compile(r'\b[A-Z]{2,10}\b'),
            # 首字母大写多词：John Doe, New York
            re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'),
            # 引号内容
            re.compile(r'["\']([^"\']{2,50})["\']'),
            # 中文实体（2-8字）
            re.compile(r'[一-鿿]{2,8}(?:公司|团队|项目|系统|平台|框架|库)'),
        ]
        # ... 提取逻辑

    def _ner_extract(self, text: str) -> list[dict]:
        """使用 spaCy 或 stanza 进行 NER。"""
        try:
            import spacy
            if not hasattr(self, '_nlp'):
                self._nlp = spacy.load("zh_core_web_sm")  # 或 en_core_web_sm
            doc = self._nlp(text)
            return [{"text": ent.text, "type": ent.label_} for ent in doc.ents]
        except (ImportError, OSError):
            return []  # 模型不可用时静默回退
```

---

## 四、验证方案

### 4.1 并发安全验证

```python
# tests/test_concurrent_memory.py
import threading
import sqlite3

def test_concurrent_read_write():
    """验证并发读写不会导致数据损坏。"""
    store = MemoryStore(db_path=":memory:")
    errors = []

    def writer():
        for i in range(100):
            try:
                store.add_fact(f"fact_{i}", category="test")
            except Exception as e:
                errors.append(e)

    def reader():
        for i in range(100):
            try:
                store.search_facts("test")
            except Exception as e:
                errors.append(e)

    threads = [threading.Thread(target=writer) for _ in range(3)]
    threads += [threading.Thread(target=reader) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent access errors: {errors}"
```

### 4.2 算法正确性验证

```python
# tests/test_salience_scoring.py
def test_no_double_counting():
    """验证重复消息不会被过度惩罚。"""
    scorer = SalienceScorer()

    # 新消息
    score1 = scorer.score("Hello world, this is important")

    # 重复消息
    score2 = scorer.score("Hello world, this is important")

    # 重复惩罚应该存在但不应过度
    assert score2 < score1, "Repetition should reduce score"
    assert score2 > score1 * 0.3, "Repetition penalty should not be extreme"

def test_new_engram_not_at_max():
    """验证新 engram 不从最大强度开始。"""
    engine = SilentEngramEngine()
    engine.strengthen("new_memory_123", delta=0.03)

    strength = engine.get_strength("new_memory_123")
    assert strength < 1.0, f"New engram should not start at max, got {strength}"
    assert 0.2 < strength < 0.5, f"New engram should start in fragile range, got {strength}"
```

### 4.3 回归测试矩阵

| 测试类别 | 测试数量 | 覆盖范围 |
|---------|---------|---------|
| 单元测试 | ~50 | 每个组件的核心逻辑 |
| 集成测试 | ~15 | Pipeline 全流程、Provider 生命周期 |
| 并发测试 | ~10 | 读写并发、锁顺序、死锁检测 |
| 性能测试 | ~5 | 1000 条记忆的 consolidation 耗时、search 延迟 |
| 中文测试 | ~10 | CJK 分词、中文情绪检测、中文实体提取 |

---

## 五、风险评估与回滚策略

### 5.1 变更风险矩阵

| Phase | 风险等级 | 回滚难度 | 关键风险 |
|-------|---------|---------|---------|
| Phase 0 | 低 | 容易 | 锁逻辑变更可能引入新死锁 |
| Phase 1 | 中 | 中等 | 算法参数变更影响记忆质量 |
| Phase 2 | 高 | 困难 | 文件拆分可能破坏导入链 |
| Phase 3 | 低 | 容易 | 新功能可选启用 |
| Phase 4 | 中 | 中等 | 新模型需要数据迁移 |

### 5.2 回滚策略

1. **Phase 0/1:** 通过 feature flag 控制新旧逻辑，`config.yaml` 中增加 `pipeline.v2_enabled: true`
2. **Phase 2:** 保留原始 `memory_pipeline.py` 作为 `memory_pipeline_legacy.py`，导入重定向
3. **Phase 3/4:** 新功能全部可选启用，默认关闭

---

## 六、总结

### 核心发现

1. **并发安全是最紧迫的问题** —— 5 个并发 bug 中 3 个是 Critical，在高负载场景下必然触发
2. **算法正确性问题导致记忆质量下降** —— 新记忆直接跳到最大强度、重复信号被双重计数、schema 信心持续通缩
3. **架构复杂度超出可维护范围** —— 2586 行单文件 + 全异常静默吞噬 = 生产环境黑盒
4. **Holographic 插件有 4 个功能完整的模块从未被使用** —— 代码已完成但集成未闭环
5. **中文支持存在系统性缺陷** —— 情绪检测、实体提取、FTS5 搜索均未针对 CJK 优化

### 优化收益预估

| 维度 | 当前状态 | 优化后预期 |
|------|---------|-----------|
| 并发安全 | 3 个 Critical 死锁/数据损坏风险 | 零已知并发 bug |
| 记忆质量 | 新记忆无脆弱期、重复惩罚过度、信心通缩 | 符合 Ebbinghaus 曲线、参数可调 |
| 可观测性 | 全异常 debug 级别 | 分级告警、连续失败自动禁用 |
| 可维护性 | 2586 行单文件 | 8 个独立模块，平均 ~200 行 |
| 中文支持 | 情绪/实体/搜索三重缺陷 | 全链路 CJK 优化 |
| 功能完整性 | 4 个死模块、2 个计算后未使用的功能 | 全部闭环 |

---

*本方案基于代码审计（34 + 22 = 56 个具体问题）和文献调研（Ebbinghaus 1885, Collins & Loftus 1975, McClelland et al. 1995, Nader et al. 2000, Anderson et al. 2004, Packer et al. 2023, Jarrett Ye 2022-2023）综合产出。*
