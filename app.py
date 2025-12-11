import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import json
import requests
import re
import base64
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Optional, Callable
import concurrent.futures
from io import BytesIO
import numpy as np  # ✅ 单独引入 numpy

# ================== 允许的测试用例类型 ==================
ALLOWED_TYPES = ["正向", "异常", "边界", "安全", "性能", "界面", "其他"]

# 每个功能点最多带入多少字符的上下文，防止 PRD 很长时每次都塞整篇
MAX_CONTEXT_CHARS = 2000

# ================== MarkMap 思维导图（可选） ==================
try:
    from streamlit_markmap import markmap
    HAS_MARKMAP = True
except ImportError:
    HAS_MARKMAP = False

# ================== 页面基础配置（⚠ 必须是第一个 st.* 调用） ==================
st.set_page_config(
    page_title="智测 AI Pro - 需求转用例工作台（强化版）",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================== 语义相似度 Embedding（可选，使用缓存） ==================
@st.cache_resource
def load_embedding_model():
    """
    只在第一次调用时加载 SentenceTransformer，后面都走缓存。
    """
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None

EMBED_MODEL = load_embedding_model()
HAS_EMBED = EMBED_MODEL is not None

# ================== JSON Repair（可选） ==================
try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except Exception:
    HAS_JSON_REPAIR = False

# ================== 通用工具函数 ==================


def clean_and_parse_json(text: str) -> Any:
    """
    强力 JSON 清洗：
    1. 直接 json.loads
    2. 使用 json_repair 修复
    3. 提取 ```json ... ``` 中间的内容
    4. 截取第一个 { 到最后一个 } 之间的内容
    """
    if not isinstance(text, str):
        raise ValueError("模型返回内容不是字符串")

    # 1. 直接解析
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2. 使用 json_repair 修复
    if HAS_JSON_REPAIR:
        try:
            repaired = repair_json(text)
            return json.loads(repaired)
        except Exception:
            pass

    # 3. ```json ... ```
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        snippet = match.group(1)
        # 先试直接 load
        try:
            return json.loads(snippet)
        except Exception:
            # 再尝试 repair
            if HAS_JSON_REPAIR:
                try:
                    repaired = repair_json(snippet)
                    return json.loads(repaired)
                except Exception:
                    pass

    # 4. 从第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end]
        try:
            return json.loads(snippet)
        except Exception:
            if HAS_JSON_REPAIR:
                try:
                    repaired = repair_json(snippet)
                    return json.loads(repaired)
                except Exception:
                    pass

    raise ValueError(f"无法提取有效 JSON，原始返回开头为: {text[:200]}...")


def get_feishu_content(url: str, app_id: str, app_secret: str) -> str:
    """
    飞书文档解析：
    - 未配置 app_id / secret 时：返回 Mock PRD 内容，保证 Demo 不翻车
    - 配置后：尝试调用飞书 API（简化版）
    - 对 Table Block 尝试转换为 Markdown 表格
    """
    if not url:
        return ""

    mock_content = """
# [模拟] B端管理后台登录功能

## 1. 账号登录
用户需输入手机号和密码。手机号需验证 11 位格式。

## 2. 异常处理
- 密码错误超过 5 次锁定账号 30 分钟。
- 网络断开时应提示“网络连接异常”。
""".strip()

    if not app_id or not app_secret:
        return f"【演示模式 - 未配置飞书 Key】\n已模拟读取文档：{url}\n\n{mock_content}"

    try:
        # 1. 获取 tenant_access_token
        token_url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
        token_resp = requests.post(
            token_url,
            json={"app_id": app_id, "app_secret": app_secret},
            timeout=15,
        ).json()

        if token_resp.get("code") != 0:
            return f"❌ 飞书鉴权失败: {token_resp.get('msg')}"

        access_token = token_resp["tenant_access_token"]

        # 2. 解析 doc_token
        doc_token = url.rstrip("/").split("/")[-1].split("?")[0]

        # 3. 获取 blocks（仅演示）
        content_url = f"https://open.feishu.cn/open-apis/docx/v1/documents/{doc_token}/blocks"
        headers = {"Authorization": f"Bearer {access_token}"}
        resp = requests.get(content_url, headers=headers, timeout=15).json()

        if resp.get("code") != 0:
            return f"❌ 文档读取失败: {resp.get('msg')}"

        full_text_lines = []

        for item in resp.get("data", {}).get("items", []):
            block_type = item.get("block_type")

            # 文本块
            if block_type == 2:
                for elem in item.get("body", {}).get("elements", []):
                    content = elem.get("text_run", {}).get("content", "")
                    if content:
                        full_text_lines.append(content)

            # 简单处理表格块：尝试转成 Markdown 表格
            elif block_type == 3:
                table = item.get("table", {})
                rows = table.get("cells") or table.get("rows") or []
                md_rows = []
                try:
                    for r in rows:
                        # 不同版本结构可能不一样，这里做尽量“防御式”的解析
                        cells = r.get("cells") if isinstance(r, dict) else r
                        row_texts = []
                        for cell in cells:
                            cell_text = ""
                            for elem in cell.get("body", {}).get("elements", []):
                                cell_text += elem.get("text_run", {}).get("content", "")
                            row_texts.append(cell_text.strip() or " ")
                        md_rows.append("| " + " | ".join(row_texts) + " |")
                    if md_rows:
                        # 简单加 header 分割线
                        if len(md_rows) >= 2:
                            col_num = md_rows[0].count("|") - 1
                            sep = "| " + " | ".join(["---"] * col_num) + " |"
                            full_text_lines.append(md_rows[0])
                            full_text_lines.append(sep)
                            full_text_lines.extend(md_rows[1:])
                        else:
                            full_text_lines.extend(md_rows)
                except Exception:
                    # 如果表格解析失败，不中断整体逻辑
                    pass

        full_text = "\n".join(full_text_lines)
        return full_text or "文档内容为空或解析失败"

    except Exception as e:
        return f"⚠️ 接口调用异常 (已切换至模拟数据): {str(e)}\n\n{mock_content}"


def call_llm(
    api_key: str,
    model_id: str,
    messages: List[Dict[str, Any]],
    response_format: Optional[Dict[str, Any]] = None,
    timeout: int = 300,
) -> str:
    """
    通用 LLM 调用封装：
    - 使用 Ark ChatCompletions
    - 默认返回 message.content 字符串
    """
    if not api_key:
        raise RuntimeError("未配置 API Key")

    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.2,
        "stream": False,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        session = requests.Session()
        session.trust_env = False
        resp = session.post(url, headers=headers, json=payload, timeout=timeout)
    except Exception as e:
        raise RuntimeError(f"调用 LLM 网络异常：{e}")

    if resp.status_code != 200:
        raise RuntimeError(f"API Error {resp.status_code}: {resp.text}")

    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ================== priority 后处理 ==================


def post_process_priority(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    简单把 priority 分层：
    - 如果模型已经给了 P0/P1/P2，就先保留；
    - 如果全部都是 P0 或非常集中，就按顺序重新划分比例。
    """
    n = len(features)
    if n == 0:
        return features

    pri_list = [f.get("priority", "P1") for f in features]
    unique_pri = set(pri_list)

    if len(unique_pri) <= 1:
        # 按序简单分档
        for idx, f in enumerate(features):
            ratio = (idx + 1) / n
            if ratio <= 0.3:
                f["priority"] = "P0"
            elif ratio <= 0.7:
                f["priority"] = "P1"
            else:
                f["priority"] = "P2"
    else:
        for f in features:
            p = f.get("priority", "P1")
            if p not in ["P0", "P1", "P2"]:
                f["priority"] = "P1"

    return features


# ================== CoT / 分治：功能点 + 分治生成 ==================


def extract_features(prd_text: str, guidelines: str, api_key: str, model_id: str) -> List[Dict[str, Any]]:
    """
    阶段一：从 PRD 中抽取功能点（features）
    返回类似：
    [{
      "id":"F1",
      "name":"登录成功",
      "desc":"...",
      "priority":"P0",
      "module":"登录模块",
      "scene_type":"正向" / "异常" / "约束" / "边界" / "安全" / "其他",
      "source_text":"来自 PRD 的关键原文片段，用于缩短后续上下文"
    }, ...]
    """
    guideline_text = guidelines.strip() or "无"

    prompt = f"""
你是一名资深测试分析师，请从以下 PRD 中抽取功能点列表，以便后续为每个功能点设计测试用例。

【重要要求】
- 你的分析和输出可以使用中文或英文，但最终 JSON 中的字段值（功能点名称、描述、模块名、scene_type、source_text）一律使用简体中文。
- JSON 的 key 使用英文（如 "id"、"name"、"scene_type"、"source_text"），value 使用中文。
- 每个功能点请增加 scene_type 字段，取值之一：
  - "正向"：正常业务主流程，如登录成功、下单成功等
  - "异常"：错误场景或异常分支，如账号不存在、密码错误、权限不足等
  - "约束"：业务约束/规则，如“用户名长度必须 1~20 位”“金额不得为负数”等
  - "边界"：专门描述边界值/临界值规则的功能点
  - "安全"：与安全、权限、风控直接相关的功能点
  - 其他情况可以用 "其他"
- 每个功能点尽量补充一个 source_text 字段：直接从 PRD 中复制与该功能点最相关的原文段落或小节，用于后续缩短上下文。

【输入】
1. PRD 文本：
{prd_text}

2. 企业测试规范（可选）：
{guideline_text}

【输出要求】
- 只输出 JSON，一定要是合法的 JSON 对象。
- JSON 格式示例：
{{
  "features": [
    {{
      "id": "F1",
      "name": "登录成功",
      "desc": "已注册用户输入正确的账号和密码登录系统并进入首页。",
      "priority": "P0",         // P0/P1/P2
      "module": "用户登录",      // 可复用 PRD 中的模块/页面名
      "scene_type": "正向",
      "source_text": "从 PRD 中复制来的相关原文"
    }}
  ]
}}
- 请尽量覆盖 PRD 中所有主要功能点（包括明显的异常/约束/边界规则），一般不超过 20 个功能点。
""".strip()

    messages = [
        {"role": "system", "content": "你是一名严谨的测试分析师，负责拆解 PRD 功能点，请用简体中文输出字段值。"},
        {"role": "user", "content": prompt},
    ]

    raw = call_llm(
        api_key=api_key,
        model_id=model_id,
        messages=messages,
        response_format={"type": "json_object"},
        timeout=180,
    )
    obj = clean_and_parse_json(raw)
    features = obj.get("features", [])
    norm_features: List[Dict[str, Any]] = []

    for idx, f in enumerate(features, start=1):
        norm_features.append(
            {
                "id": f.get("id", f"F{idx}"),
                "name": f.get("name", f"功能点 {idx}"),
                "desc": f.get("desc", ""),
                "priority": f.get("priority", "P1"),
                "module": f.get("module", f.get("name", "未分模块")),
                "scene_type": f.get("scene_type", "正向"),
                "source_text": f.get("source_text", ""),
            }
        )

    norm_features = post_process_priority(norm_features)
    return norm_features


def normalize_case_type(raw_type: str) -> str:
    """
    将模型返回的 type 归一化到 ALLOWED_TYPES 中：
    - 优先匹配中文
    - 常见英文映射
    - 其他归为 "其他"
    """
    if not raw_type:
        return "其他"
    t = str(raw_type).strip()

    if t in ALLOWED_TYPES:
        return t

    # 中文别名
    if t in ["正常", "主流程", "正向用例"]:
        return "正向"
    if t in ["异常用例", "错误", "失败场景"]:
        return "异常"
    if t in ["边界值", "边界测试"]:
        return "边界"
    if t in ["UI", "界面测试"]:
        return "界面"

    # 英文常见值
    lower = t.lower()
    if lower in ["positive", "happy path", "success"]:
        return "正向"
    if lower in ["negative", "error", "exception"]:
        return "异常"
    if "boundary" in lower:
        return "边界"
    if "security" in lower:
        return "安全"
    if "performance" in lower:
        return "性能"
    if lower in ["ui", "ux"]:
        return "界面"

    return "其他"


def normalize_cases(json_obj: Any) -> List[Dict[str, Any]]:
    """
    将模型返回的 JSON（{"cases": [...]} 或直接 [... ]）统一为：
    {
      "id": "TC-001",
      "module": "...",
      "title": "...",
      "precondition": "...",
      "steps": "...(多行)",
      "expected": "...(多行)",
      "type": one of ALLOWED_TYPES,
      "test_data": "...(JSON 或描述)",
      "post_actions": "...(清理/回滚步骤)"
    }
    """
    if isinstance(json_obj, dict) and "cases" in json_obj:
        raw_cases = json_obj["cases"]
    elif isinstance(json_obj, list):
        raw_cases = json_obj
    else:
        raise ValueError("返回 JSON 中未找到 'cases' 列表")

    norm: List[Dict[str, Any]] = []
    for idx, c in enumerate(raw_cases, start=1):
        module = c.get("module", "未分模块")
        title = c.get("title", f"未命名用例 {idx}")
        precondition = c.get("precondition", "")
        steps = c.get("steps", "")
        expected = c.get("expected", "")
        raw_type = c.get("type", "正向")
        test_data = c.get("test_data", "")
        post_actions = c.get("post_actions", "") or c.get("teardown", "")

        if isinstance(steps, list):
            steps = "\n".join(str(s) for s in steps)
        if isinstance(expected, list):
            expected = "\n".join(str(s) for s in expected)

        if isinstance(test_data, (dict, list)):
            try:
                test_data = json.dumps(test_data, ensure_ascii=False, indent=2)
            except Exception:
                test_data = str(test_data)
        else:
            test_data = str(test_data)

        if isinstance(post_actions, list):
            post_actions = "\n".join(str(s) for s in post_actions)
        else:
            post_actions = str(post_actions)

        norm.append(
            {
                "id": c.get("id", f"TC-{idx:03d}"),
                "module": module,
                "title": title,
                "precondition": str(precondition),
                "steps": str(steps),
                "expected": str(expected),
                "type": normalize_case_type(raw_type),
                "test_data": test_data,
                "post_actions": post_actions,
            }
        )
    return norm


def generate_cases_for_feature(
    feature: Dict[str, Any],
    prd_text: str,
    guidelines: str,
    api_key: str,
    model_id: str,

) -> List[Dict[str, Any]]:
    """
    阶段二：针对单个功能点生成用例
    - 引入企业测试规范（guidelines）
    - 输出 JSON: {"cases":[...]}
    - 所有字段值要求用简体中文
    - test_data/post_actions 字段：用于后续自动化测试/清理
    - 根据 scene_type 区分策略：
      - 正向：主流程 + 异常 + 边界（如果有）
      - 异常/约束/边界：聚焦异常和边界，不强行造无关正向
    """
    guideline_text = guidelines.strip() or "无"
    scene_type = feature.get("scene_type", "正向")

    # 为了降低上下文长度，优先使用功能点自带的 source_text，并做截断
    raw_context = feature.get("source_text") or prd_text
    context_text = raw_context[:MAX_CONTEXT_CHARS]




    if scene_type == "异常":
        coverage_text = """
    本功能点本身是异常类功能点（例如“未注册用户登录失败”）。
    请围绕该异常场景设计合适数量的用例：
    - 如果场景比较简单，可以只设计 1~2 条典型用例；
    - 如果存在多种错误类型、不同用户状态或明显边界情况，可以适当多写几条（例如 3~5 条）；
    - 至少要保证有 1 条能代表该异常场景的用例。
    不需要为该功能点额外生成“用户名密码均正确时登录成功”之类的正向用例。
    """
    elif scene_type in ("约束", "边界"):
        coverage_text = """
    本功能点属于约束/边界类功能点（例如“用户名长度必须在 1~20 位以内”）。
    请围绕该约束/边界设计合适数量的用例：
    - 至少 1 条用例体现边界内合法值的成功场景（例如长度刚好等于最小/最大值时操作成功）；
    - 可以根据复杂度，增加 1~3 条超出边界的失败场景（例如长度为 0 或大于最大限制时操作失败）；
    - 若某个场景同时是边界又是异常，只需写一条用例，并优先将 type 标记为“边界”，不要为同一场景重复生成两条。
    """
    else:
        coverage_text = f"""
    本功能点属于正常业务主流程功能点（scene_type="{scene_type}"）。
    请围绕该功能点设计合适数量的用例：
    - 至少 1 条核心正向流程用例（例如：输入合法参数后操作成功）；
    - 可以根据功能复杂度，增加若干典型异常场景（如必填项为空、格式错误、权限不足等）；
    - 如有明显边界值（长度/范围），建议至少包含 1 条边界用例；
    - 如规范中提到安全/界面要求，可增加对应 type 为“安全”或“界面”的用例。
    如果某个场景同时既是异常又是边界（例如“长度超过最大值时校验失败”），请只写一条用例，并优先将 type 标记为“边界”，不要为同一场景重复生成两条。
    """

    prompt = f"""
你是一名资深测试工程师，请针对一个具体功能点设计测试用例。

【重要要求】
- 所有字段内容（module/title/precondition/steps/expected/type/test_data/post_actions 等）一律使用简体中文。
- type 字段的取值尽量使用以下枚举之一：{ALLOWED_TYPES}。
- test_data 字段用于描述本用例所需的测试数据，可以是结构化 JSON 字符串（例如：{{"username":"test_user","password":"123456"}}）或自然语言描述。
- post_actions 字段用于描述测试结束后的清理/回滚操作，例如“删除测试账号”“还原配置”。
- JSON 的 key 使用英文，value 使用中文。
- 不要输出任何解释性文字，只能输出 JSON 对象。

【功能点信息】
{json.dumps(feature, ensure_ascii=False, indent=2)}

【与本功能点最相关的 PRD 原文片段】
{context_text}

【企业测试规范（可选）】
{guideline_text}

【用例设计策略】（请严格遵守）
{coverage_text}

【输出格式】
只输出 JSON 对象，格式如下：
{{
  "cases": [
    {{
      "id": "TC-001",
      "module": "{feature.get('module','')}",
      "title": "用例标题（中文）",
      "precondition": "前置条件（中文，可为空）",
      "steps": [
        "步骤1（中文）",
        "步骤2（中文）"
      ],
      "expected": [
        "预期结果1（中文）",
        "预期结果2（中文）"
      ],
      "type": "正向",        // 例如：正向 / 异常 / 边界 / 安全 / 性能 / 界面 / 其他
      "test_data": "测试数据描述或 JSON 字符串",
      "post_actions": "清理/回滚操作描述（可为空字符串）"
    }}
  ]
}}
""".strip()

    messages = [
        {"role": "system", "content": "你是一名严谨的测试工程师，请用简体中文编写测试用例。"},
        {"role": "user", "content": prompt},
    ]

    raw = call_llm(
        api_key=api_key,
        model_id=model_id,
        messages=messages,
        response_format={"type": "json_object"},
        timeout=240,
    )
    obj = clean_and_parse_json(raw)
    cases = normalize_cases(obj)

    for c in cases:
        c["featureId"] = feature["id"]
    return cases


def normalize_title_for_dedup(title: str) -> str:
    """
    标题归一化用于去重：
    - 去掉“（边界值）”“[边界]”等标记
    - 去掉空白
    """
    t = title or ""
    t = t.replace("（边界值）", "")
    t = t.replace("(边界值)", "")
    t = t.replace("[边界]", "")
    t = re.sub(r"\s+", "", t)
    return t


def semantic_dedup_cases(cases: List[Dict[str, Any]], sim_threshold: float = 0.85) -> List[Dict[str, Any]]:
    """
    语义去重：
    - 同一 module 内，如果两条用例的 (title+steps) 余弦相似度 > sim_threshold，则认为场景重复
    - 保留描述更详细（steps+expected 更长）的那一条
    """
    if not HAS_EMBED or EMBED_MODEL is None:
        return cases

    texts = [
        (idx, c.get("module", ""), (c.get("title", "") or "") + " " + (c.get("steps", "") or ""))
        for idx, c in enumerate(cases)
    ]
    if not texts:
        return cases

    indices, modules, contents = zip(*texts)  # type: ignore
    try:
        emb = EMBED_MODEL.encode(list(contents), convert_to_numpy=True)
    except Exception:
        return cases

    n = len(cases)
    keep = [True] * n

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            if modules[i] != modules[j]:
                continue
            va = emb[i]
            vb = emb[j]
            denom = (np.linalg.norm(va) + 1e-8) * (np.linalg.norm(vb) + 1e-8)
            sim = float(np.dot(va, vb) / denom)
            if sim >= sim_threshold:
                # 比较 steps+expected 长度，保留更详细的那条
                ci = cases[i]
                cj = cases[j]
                len_i = len(ci.get("steps", "")) + len(ci.get("expected", ""))
                len_j = len(cj.get("steps", "")) + len(cj.get("expected", ""))
                if len_i >= len_j:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

    return [c for idx, c in enumerate(cases) if keep[idx]]


def generate_test_cases_pipeline(
    prd_text: str,
    guidelines: str,
    api_key: str,
    model_id: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,  # 新增，用于更新进度条
    enable_semantic_dedup: bool = False,  # 新增：是否开启语义去重
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

    """
    整体生成流程（分治版）：
    1. 抽取功能点 features
    2. 针对每个功能点，让模型自行判断需要多少条用例（至少 1 条）
    3. 按功能点逐个生成用例
    4. 去重（合并异常 + 边界重复）
    """
    features = extract_features(prd_text, guidelines, api_key, model_id)
    if not features:
        raise RuntimeError("未能从 PRD 中抽取到功能点，无法生成用例。")

    all_cases: List[Dict[str, Any]] = []
    total = len(features)

    # ✅ 使用线程池并发为每个功能点生成用例
    # 可根据自己接口限流情况调整 max_workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, total)) as executor:
        future_to_feature = {
            executor.submit(
                generate_cases_for_feature,
                f,
                prd_text,
                guidelines,
                api_key,
                model_id,

            ): f
            for f in features
        }

        done = 0
        for future in concurrent.futures.as_completed(future_to_feature):
            f = future_to_feature[future]
            try:
                cases_f = future.result()
                all_cases.extend(cases_f)
            except Exception as e:
                print(f"为功能点 {f['id']} 生成用例失败：{e}")
            finally:
                done += 1
                if progress_callback is not None:
                    progress_callback(done, total)

    # 🔁 先做一次简单的“模块 + 归一化标题”去重
    seen = {}
    dedup_cases: List[Dict[str, Any]] = []

    for c in all_cases:
        raw_title = c.get("title", "")
        norm_title = normalize_title_for_dedup(raw_title)
        key = (c.get("module", ""), norm_title)

        if key in seen:
            old_idx = seen[key]
            old_type = dedup_cases[old_idx]["type"]
            new_type = c.get("type", old_type)
            # 如果旧的是“异常”，新的是“边界”，我们用边界覆盖异常
            if old_type == "异常" and new_type == "边界":
                dedup_cases[old_idx]["type"] = "边界"
            continue

        seen[key] = len(dedup_cases)
        dedup_cases.append(c)

    # ⭐ 第二层：可选的语义相似去重（Embedding）
    if enable_semantic_dedup and HAS_EMBED and EMBED_MODEL is not None:
        dedup_cases = semantic_dedup_cases(dedup_cases, sim_threshold=0.85)

    return features, dedup_cases




# ================== 快速模式：单轮生成 ==================


def generate_test_cases_quick(
    prd_text: str,
    guidelines: str,
    api_key: str,
    model_id: str,
    max_cases: int = 50,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    快速模式：一次性调用大模型生成测试用例，不做功能点拆解。
    返回值形式与 pipeline 一致： (features, cases)
    features 这里先返回空列表 []。
    """
    guideline_text = guidelines.strip() or "无"

    prompt = f"""
你是一名资深测试工程师，请根据下面的 PRD 内容直接生成测试用例。

【重要要求】
- 所有字段内容（module/title/precondition/steps/expected/type/test_data/post_actions 等）一律使用简体中文。
- type 字段的取值尽量使用以下枚举之一：{ALLOWED_TYPES}。
- JSON 的 key 使用英文（如 "title"、"steps"），value 必须是中文。
- test_data 字段用于描述本用例所需的测试数据，可以是结构化 JSON 字符串（例如：{{"username":"test_user","password":"123456"}}）或自然语言描述。
- post_actions 字段用于描述测试结束后的清理/回滚操作，例如“删除测试账号”“还原配置”。
- 不要输出任何解释性文字，只能输出 JSON 对象。

【PRD 内容】
{prd_text}

【企业测试规范（可选）】
{guideline_text}

【任务要求】
- 直接根据整个 PRD 设计测试用例，数量控制在不超过 {max_cases} 条。
- 覆盖：主要正向流程、典型异常场景、重要边界场景和关键安全/界面要求（如果规范中有提到）。
- 每条用例只测试一个清晰的场景。

【输出格式】
只输出 JSON 对象，格式如下：
{{
  "cases": [
    {{

      "id": "TC-001",
      "module": "模块名称（中文）",
      "title": "用例标题（中文）",
      "precondition": "前置条件（中文，可为空字符串）",
      "steps": ["步骤1（中文）", "步骤2（中文）"],
      "expected": ["预期结果1（中文）", "预期结果2（中文）"],
      "type": "正向",           // 正向 / 异常 / 边界 / 安全 / 性能 / 界面 / 其他
      "test_data": "测试数据描述或 JSON 字符串",
      "post_actions": "清理/回滚操作描述（可为空字符串）"
    }}
  ]
}}
""".strip()

    messages = [
        {
            "role": "system",
            "content": "你是一名能够快速产出高质量测试用例的资深测试工程师，请始终使用简体中文编写用例内容。",
        },
        {"role": "user", "content": prompt},
    ]

    raw = call_llm(
        api_key=api_key,
        model_id=model_id,
        messages=messages,
        response_format={"type": "json_object"},
        timeout=240,
    )
    obj = clean_and_parse_json(raw)
    cases = normalize_cases(obj)

    return [], cases


# ================== 评测相关函数 ==================


def compute_basic_metrics(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """格式合规率 + 冗余度 + 模糊词数量"""
    total = len(cases)
    if total == 0:
        return {
            "format_rate": 0.0,
            "valid_cases": 0,
            "redundancy": 0.0,
            "unique_titles": 0,
            "vague_count": 0,
        }

    valid_count = 0
    titles = set()
    vague_words = ["等等", "大概", "可能", "左右", "相关"]
    vague_count = 0

    for c in cases:
        title = (c.get("title") or "").strip()
        steps = (c.get("steps") or "").strip()
        expected = (c.get("expected") or "").strip()

        if title and steps and expected:
            valid_count += 1

        if title:
            titles.add((c.get("module", "") + "::" + title.lower()).strip())

        content = steps + " " + expected
        for w in vague_words:
            if w in content:
                vague_count += 1

    format_rate = valid_count / total
    unique_titles = len(titles)
    redundancy = 0.0
    if total > 0:
        redundancy = max(0.0, 1.0 - unique_titles / total)

    return {
        "format_rate": format_rate,
        "valid_cases": valid_count,
        "redundancy": redundancy,
        "unique_titles": unique_titles,
        "vague_count": vague_count,
    }


def jaccard_similarity(a: str, b: str) -> float:
    """超简版 Jaccard（字符集合），作为兜底"""
    set_a = set(a)
    set_b = set(b)
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def embedding_title_similarity(ai_cases: List[Dict[str, Any]], human_titles: List[str]) -> float:
    """
    语义相似度（可选）：基于 SentenceTransformer 的标题向量相似度
    - 对每条人工标题，在 AI 标题中找到最高 cosine，相加取平均
    """
    if not HAS_EMBED or EMBED_MODEL is None:
        raise RuntimeError("当前环境未安装 sentence-transformers 或模型加载失败。")

    ai_titles = [c.get("title", "") for c in ai_cases if c.get("title")]
    if not ai_titles or not human_titles:
        return 0.0

    ai_emb = EMBED_MODEL.encode(ai_titles, convert_to_numpy=True)
    human_emb = EMBED_MODEL.encode(human_titles, convert_to_numpy=True)

    sims = []
    for h in human_emb:
        denom = np.linalg.norm(ai_emb, axis=1) * (np.linalg.norm(h) + 1e-8)
        scores = np.dot(ai_emb, h) / (denom + 1e-8)
        sims.append(float(scores.max()))
    return float(np.mean(sims)) if sims else 0.0


def evaluate_against_human_csv(ai_cases: List[Dict[str, Any]], human_df: pd.DataFrame) -> Dict[str, float]:
    """
    CSV/Excel 人工用例对比：
    - 必须有 'title' 列
    - 返回：{"jaccard":..., "semantic":..., "recall":..., "precision":..., "f1":...}
    """
    if "title" not in human_df.columns:
        raise RuntimeError("人工用例 CSV/Excel 中必须包含列名为 'title' 的列")

    human_titles = [str(t) for t in human_df["title"].tolist() if str(t).strip()]
    if not human_titles:
        return {"jaccard": 0.0, "semantic": 0.0, "recall": 0.0, "precision": 0.0, "f1": 0.0}

    ai_titles = [c.get("title", "") for c in ai_cases if c.get("title")]
    ai_concat = "".join(ai_titles)
    human_concat = "".join(human_titles)
    jac = jaccard_similarity(ai_concat, human_concat) * 100

    sem = 0.0
    recall = precision = f1 = 0.0
    if HAS_EMBED and EMBED_MODEL is not None and ai_titles:
        try:
            ai_emb = EMBED_MODEL.encode(ai_titles, convert_to_numpy=True)
            human_emb = EMBED_MODEL.encode(human_titles, convert_to_numpy=True)

            # 对每个人工 title，在 AI 中找最高相似度
            hit_h = 0
            for h_vec in human_emb:
                denom = np.linalg.norm(ai_emb, axis=1) * (np.linalg.norm(h_vec) + 1e-8)
                scores = np.dot(ai_emb, h_vec) / (denom + 1e-8)
                if scores.max() >= 0.75:
                    hit_h += 1
            recall = hit_h / len(human_titles) if human_titles else 0.0

            # 对每个 AI title，在人工中找最高相似度
            hit_ai = 0
            for a_vec in ai_emb:
                denom = np.linalg.norm(human_emb, axis=1) * (np.linalg.norm(a_vec) + 1e-8)
                scores = np.dot(human_emb, a_vec) / (denom + 1e-8)
                if scores.max() >= 0.75:
                    hit_ai += 1
            precision = hit_ai / len(ai_titles) if ai_titles else 0.0

            if recall + precision > 0:
                f1 = 2 * recall * precision / (recall + precision)

            # 语义相似度：人工标题对 AI 标题的平均最高相似度
            sims = []
            for h_vec in human_emb:
                denom = np.linalg.norm(ai_emb, axis=1) * (np.linalg.norm(h_vec) + 1e-8)
                scores = np.dot(ai_emb, h_vec) / (denom + 1e-8)
                sims.append(float(scores.max()))
            sem = float(sum(sims) / len(sims)) if sims else 0.0
        except Exception:
            sem = 0.0

    return {
        "jaccard": jac,
        "semantic": sem * 100,
        "recall": recall * 100,
        "precision": precision * 100,
        "f1": f1 * 100,
    }


def judge_by_llm(
    api_key: str,
    model_id: str,
    prd_text: str,
    cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    LLM-as-a-Judge：
    输入 PRD + 用例，输出：
      { completeness_score, clarity_score, overall_score, comments }
    """
    if not cases:
        raise RuntimeError("没有生成的用例，无法评审")

    short_cases = [
        {
            "id": c["id"],
            "module": c["module"],
            "title": c["title"],
            "type": c["type"],
            "steps": c["steps"],
            "expected": c["expected"],
        }
        for c in cases
    ]

    prompt = f"""
你现在是一名非常严格的测试经理，需要对一批由大模型生成的测试用例进行质量评审。

【PRD 内容】
{prd_text}

【测试用例列表（关键信息）】
{json.dumps(short_cases, ensure_ascii=False, indent=2)}

【请给出如下评分（0~10，支持一位小数）】：
1. completeness_score：完整性。是否覆盖了 PRD 主要功能点以及重要的异常/边界场景？（可看作需求覆盖率的近似）
2. clarity_score：清晰度。用例描述是否清晰、具体、可执行？是否存在大量模糊表述？
3. overall_score：综合评分。综合考虑完整性、清晰度、数量、冗余等后的总体评价。

【输出要求】
只输出一个 JSON 对象，例如：
{{
  "completeness_score": 8.5,
  "clarity_score": 9.0,
  "overall_score": 8.8,
  "comments": "这里写你对这批用例的总体评价和改进建议。"
}}
""".strip()

    messages = [
        {"role": "system", "content": "你是一名严谨的测试经理，负责评审测试用例质量。"},
        {"role": "user", "content": prompt},
    ]

    raw = call_llm(
        api_key=api_key,
        model_id=model_id,
        messages=messages,
        response_format={"type": "json_object"},
        timeout=240,
    )
    obj = clean_and_parse_json(raw)
    return obj


def coverage_by_llm(
    api_key: str,
    model_id: str,
    prd_text: str,
    features: List[Dict[str, Any]],
    cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    使用 LLM 检查“功能点覆盖率”：
    - 输入：功能点列表 + 用例列表
    - 输出：{coverage_score, uncovered_features, analysis}
    """
    if not features:
        raise RuntimeError("当前没有功能点列表，无法进行覆盖率分析。")

    short_features = [
        {"id": f["id"], "name": f["name"], "desc": f.get("desc", "")}
        for f in features
    ]
    short_cases = [
        {"id": c["id"], "module": c["module"], "title": c["title"], "type": c["type"]}
        for c in cases
    ]

    prompt = f"""
你是一名资深测试经理，需要从“功能点覆盖”的角度检查测试用例是否完整。

【PRD 内容】
{prd_text}

【功能点列表】
{json.dumps(short_features, ensure_ascii=False, indent=2)}

【测试用例列表（简版）】
{json.dumps(short_cases, ensure_ascii=False, indent=2)}

【任务】
- 请你逐个检查功能点，判断是否至少有一条测试用例可以覆盖该功能点。
- 如果某个功能点完全没有被任何用例覆盖，请将它记为“未覆盖功能点”。

【输出要求】
只输出一个 JSON 对象，字段包括：
{{
  "coverage_score": 0.85,            // 覆盖率 = 1 - 未覆盖功能点数 / 功能点总数
  "uncovered_features": ["F2","F5"], // 未覆盖的功能点 id 列表（如没有则为空数组）
  "analysis": "对覆盖情况的简要分析和改进建议（中文）"
}}
""".strip()

    messages = [
        {"role": "system", "content": "你是一名关注需求覆盖率的测试经理。"},
        {"role": "user", "content": prompt},
    ]

    raw = call_llm(
        api_key=api_key,
        model_id=model_id,
        messages=messages,
        response_format={"type": "json_object"},
        timeout=240,
    )
    obj = clean_and_parse_json(raw)
    return obj


def hallucination_check_by_llm(
    api_key: str,
    model_id: str,
    prd_text: str,
    cases: List[Dict[str, Any]],
    max_cases_check: int = 20,
) -> Dict[str, Any]:
    """
    幻觉检测：
    - 抽样若干条用例，让 LLM 判断“预期结果”是否有 PRD 依据
    - 返回：{ suspicious_cases: [...], summary: "..." }
    """
    if not cases:
        raise RuntimeError("没有用例，无法进行幻觉检测。")

    sample_cases = cases[:max_cases_check]
    short_cases = [
        {
            "id": c["id"],
            "module": c["module"],
            "title": c["title"],
            "steps": c["steps"],
            "expected": c["expected"],
        }
        for c in sample_cases
    ]

    prompt = f"""
你是一名非常严谨的需求分析师，需要检查以下由大模型生成的测试用例是否存在“幻觉”——即预期结果中包含了 PRD 中并未提到的逻辑。

【PRD 内容】
{prd_text}

【待检查的测试用例（抽样）】
{json.dumps(short_cases, ensure_ascii=False, indent=2)}

【任务】
- 对每条用例，检查其“预期结果”是否可以在 PRD 中找到依据。
- 如果预期结果与 PRD 描述不符，或者 PRD 根本没有提到相关逻辑，则判定该用例为“疑似幻觉”。
- 注意：统一错误文案等细节可以适当宽松，但不能凭空出现新的业务规则或流程。

【输出要求】
只输出一个 JSON 对象，例如：
{{
  "suspicious_cases": [
    {{"id":"TC-003","reason":"预期提到了账号锁定规则，但 PRD 中没有相关描述"}}
  ],
  "summary": "整体幻觉比例较低，大部分用例预期都有 PRD 依据。"
}}
""".strip()

    messages = [
        {"role": "system", "content": "你是一名负责发现大模型幻觉问题的需求分析师。"},
        {"role": "user", "content": prompt},
    ]

    raw = call_llm(
        api_key=api_key,
        model_id=model_id,
        messages=messages,
        response_format={"type": "json_object"},
        timeout=300,
    )
    obj = clean_and_parse_json(raw)
    return obj


def improve_cases_with_llm(
    api_key: str,
    model_id: str,
    prd_text: str,
    guidelines: str,
    cases: List[Dict[str, Any]],
    judge_result: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Self-Correction：根据评审意见自动优化用例
    - 输入：原始 PRD、企业测试规范、当前用例、评审意见（可为空）
    - 输出：新的 {"cases":[...]}，字段结构与原始用例一致
    """
    guideline_text = guidelines.strip() or "无"
    judge_text = json.dumps(judge_result, ensure_ascii=False, indent=2) if judge_result else "暂无评审意见"

    prompt = f"""
你是一名资深测试专家，现在需要根据评审意见，对一批自动生成的测试用例进行“二次优化”。

【重要要求】
- 所有字段内容（module/title/precondition/steps/expected/type/test_data/post_actions 等）一律使用简体中文。
- type 字段的取值尽量使用以下枚举之一：{ALLOWED_TYPES}。
- JSON 的 key 使用英文，value 使用中文。
- 不要输出任何解释性文字，只能输出 JSON 对象。

【PRD 内容】
{prd_text}

【企业测试规范】
{guideline_text}

【当前测试用例列表】
{json.dumps(cases, ensure_ascii=False, indent=2)}

【评审意见（来自测试经理或 LLM）】
{judge_text}

【任务】
- 在保持字段结构不变的前提下（id/module/title/precondition/steps/expected/type/test_data/post_actions），对测试用例进行优化：
  - 可以对用例标题、步骤、预期进行改写，使其更清晰、具体、可执行；
  - 可以删除明显冗余的用例（重复测试同一场景且没有边界差异）；
  - 可以增加少量关键的异常/边界/安全/界面场景用例；
  - 尽量提升完整性和清晰度，同时控制冗余度。
- 最终输出一批新的测试用例，数量与当前用例大致相当（不必完全相等）。

【输出格式】
只输出 JSON 对象，格式为：
{{
  "cases": [
    {{
      "id": "TC-001",
      "module": "...",
      "title": "...",
      "precondition": "...",
      "steps": ["..."],
      "expected": ["..."],
      "type": "正向",
      "test_data": "测试数据描述或 JSON 字符串",
      "post_actions": "清理/回滚操作描述（可为空字符串）"
    }}
  ]
}}
""".strip()

    messages = [
        {"role": "system", "content": "你是一名能够根据评审意见自动优化测试用例的测试专家。"},
        {"role": "user", "content": prompt},
    ]

    raw = call_llm(
        api_key=api_key,
        model_id=model_id,
        messages=messages,
        response_format={"type": "json_object"},
        timeout=300,
    )
    obj = clean_and_parse_json(raw)
    new_cases = normalize_cases(obj)
    return new_cases


def build_markdown_cases(cases: List[Dict[str, Any]]) -> str:
    """导出 Markdown 版测试用例"""
    lines: List[str] = []
    module_map: Dict[str, List[Dict[str, Any]]] = {}
    for c in cases:
        module_map.setdefault(c["module"], []).append(c)

    for module, group in module_map.items():
        lines.append(f"## 模块：{module}")
        lines.append("")
        for c in group:
            lines.append(f"### {c['id']} - {c['title']}")
            lines.append("")
            lines.append(f"**类型：** {c['type']}")
            lines.append("")
            if c.get("test_data"):
                lines.append("**测试数据：**")
                lines.append(c["test_data"])
                lines.append("")
            lines.append("**前置条件：**")
            lines.append(c["precondition"] or "（无）")
            lines.append("")
            lines.append("**操作步骤：**")
            for i, s in enumerate((c["steps"] or "").splitlines(), start=1):
                lines.append(f"{i}. {s}")
            lines.append("")
            lines.append("**预期结果：**")
            for i, s in enumerate((c["expected"] or "").splitlines(), start=1):
                lines.append(f"{i}. {s}")
            lines.append("")
            if c.get("post_actions"):
                lines.append("**后置处理 / 清理：**")
                for i, s in enumerate((c["post_actions"] or "").splitlines(), start=1):
                    lines.append(f"{i}. {s}")
                lines.append("")
    return "\n".join(lines)


def build_markmap_md(cases: List[Dict[str, Any]]) -> str:
    """
    MarkMap 思维导图 Markdown：
    # 测试用例结构
    - 模块
      - 用例ID + 标题
        - 步骤
        - 预期
    """
    lines: List[str] = ["# 测试用例结构"]

    module_map: Dict[str, List[Dict[str, Any]]] = {}
    for c in cases:
        module_map.setdefault(c.get("module", "未分模块"), []).append(c)

    for module, group in module_map.items():
        lines.append(f"- {module}")
        for c in group:
            title = f"{c.get('id','')} {c.get('title','')}".strip()
            lines.append(f"  - {title}")
            for s in (c.get("steps", "") or "").splitlines()[:3]:
                s = s.strip()
                if s:
                    lines.append(f"    - 步骤：{s}")
            for e in (c.get("expected", "") or "").splitlines()[:1]:
                e = e.strip()
                if e:
                    lines.append(f"    - 预期：{e}")

    return "\n".join(lines)


# ================== 侧边栏配置 ==================

with st.sidebar:
    st.header("⚙️ 大模型参数")
    ark_api_key = st.text_input(
        "火山引擎 API Key",
        type="password",
        key="ark_api_key",
        value=st.session_state.get("ark_api_key", ""),
    )
    model_id = st.text_input(
        "生成模型 ID",
        value=st.session_state.get("model_id", "doubao-seed-1-6-251015"),
        key="model_id",
    )
    judge_model_id = st.text_input(
        "评审 / 自我修正模型 ID（可选）",
        value=st.session_state.get("judge_model_id", "deepseek-r1-250528"),
        key="judge_model_id",
    )


    st.divider()
    st.header("🏢 企业测试规范（RAG 思想）")
    test_guidelines = st.text_area(
        "测试规范 / 内部质量标准（可选）",
        height=150,
        placeholder="例如：\n- 所有密码传输必须加密\n- 金额字段不得为负数\n- 管理员操作需记录审计日志\n...",
    )

    st.divider()
    with st.expander("飞书配置 (加分项)", expanded=False):
        fs_app_id = st.text_input("Feishu App ID (可选)")
        fs_secret = st.text_input("Feishu App Secret (可选)", type="password")
        st.caption("不填则使用 Mock PRD 内容用于演示。")

# ================== 全局状态 ==================

if "prd_text" not in st.session_state:
    st.session_state["prd_text"] = ""
if "features" not in st.session_state:
    st.session_state["features"] = []
if "cases" not in st.session_state:
    st.session_state["cases"] = []
if "ui_image_b64" not in st.session_state:
    st.session_state["ui_image_b64"] = None
if "judge_result" not in st.session_state:
    st.session_state["judge_result"] = None
if "coverage_result" not in st.session_state:
    st.session_state["coverage_result"] = None
if "hallucination_result" not in st.session_state:
    st.session_state["hallucination_result"] = None

# ================== 页面布局 ==================

st.title("🧬 智测 AI Pro - 需求驱动测试用例生成 & 自我优化平台")

tab1, tab2, tab3 = st.tabs(["📄 需求输入", "🚀 用例生成 & 可视化", "📊 效果评测 & 自我修正"])

# -------- Tab1: 需求输入 --------
with tab1:
    col_in1, col_in2 = st.columns([2, 1])

    with col_in1:
        input_method = st.radio("选择输入来源", ["文本粘贴", "飞书链接解析"], horizontal=True)
        if input_method == "文本粘贴":
            prd_text_input = st.text_area(
                "需求文档内容",
                value=st.session_state["prd_text"],
                height=320,
                placeholder="请在此输入 PRD 文本（支持 Markdown）...",
            )
            st.session_state["prd_text"] = prd_text_input
        else:
            fs_url = st.text_input("飞书文档链接")
            if st.button("🔍 解析飞书文档"):
                with st.spinner("正在调用飞书 API 或使用 Mock 数据..."):
                    content = get_feishu_content(fs_url, fs_app_id, fs_secret)
                    st.session_state["prd_text"] = content
                    st.success("文档解析完成，已写入输入框")
            st.text_area("解析后的 PRD 内容", st.session_state["prd_text"], height=320)

    with col_in2:
        st.markdown("#### 📸 UI 辅助生成（多模态，可选）")
        uploaded_file = st.file_uploader("上传 UI 设计图（PNG/JPG）", type=["png", "jpg", "jpeg"])
        if uploaded_file:
            st.image(uploaded_file, caption="已启用视觉增强（目前仅加入 Prompt）", use_column_width=True)
            st.session_state["ui_image_b64"] = base64.b64encode(uploaded_file.getvalue()).decode()
        else:
            st.session_state["ui_image_b64"] = None

# -------- Tab2: 用例生成 & 可视化 --------
with tab2:
    st.subheader("2.1 测试用例生成")

    mode = st.radio(
        "生成模式",
        ["快速模式（单轮生成）", "精细模式（CoT+分治）"],
        horizontal=True,
    )

    if st.button("开始生成测试用例", type="primary"):
        prd_text = st.session_state["prd_text"]
        if not ark_api_key:
            st.error("请先在侧边栏配置火山引擎 API Key")
        elif not prd_text.strip():
            st.warning("请先在 Tab1 中输入或解析 PRD 内容")
        else:
            if mode.startswith("快速模式"):
                with st.spinner("🤖 正在快速生成测试用例..."):
                    try:
                        features, cases = generate_test_cases_quick(
                            prd_text=prd_text,
                            guidelines=test_guidelines,
                            api_key=ark_api_key,
                            model_id=model_id,
                            max_cases=50,
                        )
                        st.session_state["features"] = features
                        st.session_state["cases"] = cases
                        st.success(f"✅ [快速模式] 已生成 {len(cases)} 条测试用例")
                    except Exception as e:
                        st.error(f"生成过程出错：{e}")
            else:
                # 精细模式：增加进度条，显示每个功能点的完成情况
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def _progress_cb(done: int, total: int):
                    ratio = done / max(total, 1)
                    progress_bar.progress(ratio)
                    status_text.text(f"已完成 {done}/{total} 个功能点的用例生成...")

                with st.spinner("🤖 正在进行功能点拆解并并发生成测试用例..."):
                    try:
                        features, cases = generate_test_cases_pipeline(
                            prd_text=prd_text,
                            guidelines=test_guidelines,
                            api_key=ark_api_key,
                            model_id=model_id,

                        )
                        st.session_state["features"] = features
                        st.session_state["cases"] = cases

                        if features:
                            avg_per_feature = len(cases) / max(len(features), 1)
                            st.success(
                                f"✅ [精细模式] 已为 {len(features)} 个功能点生成 {len(cases)} 条测试用例 "
                                f"(约 {avg_per_feature:.1f} 条/功能点)"
                            )
                        else:
                            st.success(f"✅ [精细模式] 已生成 {len(cases)} 条测试用例")
                    except Exception as e:
                        st.error(f"生成过程出错：{e}")

    features = st.session_state["features"]
    cases = st.session_state["cases"]

    if features:
        st.markdown("### 2.2 功能点列表（CoT 抽取结果）")
        df_features = pd.DataFrame(features)
        st.dataframe(df_features, use_container_width=True, height=220)

    if not cases:
        st.info("尚未生成用例，请先点击上方按钮。")
    else:
        st.markdown("### 2.3 用例表格视图（支持直接编辑）")

        df_cases = pd.DataFrame(cases)
        edited_df = st.data_editor(
            df_cases,
            column_config={
                "type": st.column_config.SelectboxColumn(
                    "类型",
                    options=ALLOWED_TYPES,
                    width="small",
                ),
                "steps": st.column_config.TextColumn("步骤（可多行）"),
                "expected": st.column_config.TextColumn("预期结果（可多行）"),
                "test_data": st.column_config.TextColumn("测试数据（JSON 或描述）"),
                "post_actions": st.column_config.TextColumn("后置处理 / 清理"),
            },
            use_container_width=True,
            num_rows="dynamic",
        )
        st.session_state["cases"] = edited_df.to_dict(orient="records")

        st.markdown("#### 导出用例")
        csv_bytes = edited_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 导出 CSV",
            data=csv_bytes,
            file_name="testcases.csv",
            mime="text/csv",
        )
        md_str = build_markdown_cases(st.session_state["cases"])
        st.download_button(
            "📥 导出 Markdown（详细用例）",
            data=md_str.encode("utf-8"),
            file_name="testcases.md",
            mime="text/markdown",
        )

        # Excel 导出
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            edited_df.to_excel(writer, index=False, sheet_name="TestCases")
            # 可以在此处进一步设置单元格换行、列宽等
        excel_buffer.seek(0)
        st.download_button(
            "📥 导出 Excel (.xlsx)",
            data=excel_buffer,
            file_name="testcases.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.markdown("---")
        st.markdown("### 2.4 用例结构思维导图（MarkMap MindMap）")

        if HAS_MARKMAP:
            mm_md = build_markmap_md(st.session_state["cases"])
            markmap(mm_md, height=450)
            st.caption("提示：在图中可以用鼠标拖动节点、滚轮缩放查看整体结构。")
        else:
            st.info("未安装 streamlit-markmap，如需脑图请执行：`pip install streamlit-markmap` 后重启。")

# -------- Tab3: 效果评测 & 自我修正 --------
with tab3:
    st.header("📊 效果评测 & 🔁 自我修正（Self-Correction）")

    cases = st.session_state["cases"]
    prd_text = st.session_state["prd_text"]
    features = st.session_state["features"]

    if not cases:
        st.info("尚未生成用例，请在 Tab2 中先生成。")
    else:
        col_eval_left, col_eval_right = st.columns([1, 2])

        with col_eval_left:
            st.markdown("#### 人工基准输入（可选）")
            golden_text = st.text_area(
                "人工标准用例（纯文本，用于粗略相似度）",
                height=180,
                placeholder="可粘贴人工写的用例标题或摘要，不填则跳过此项。",
            )

            st.markdown("#### 上传人工用例 CSV/Excel（可选）")
            st.caption("文件中需至少包含 `title` 列。")
            uploaded_gold = st.file_uploader("上传人工用例文件", type=["csv", "xlsx", "xls"])

            run_eval = st.button("⚖️ 规则+统计评估", type="primary")
            run_judge = st.button("🧠 使用 LLM 评审")
            run_coverage = st.button("🔎 功能点覆盖检查")
            run_hallu = st.button("🧯 幻觉检查")
            run_improve = st.button("🔁 根据评审意见自动优化用例")

        with col_eval_right:
            df_cases = pd.DataFrame(cases)

            # 1. 场景类型分布
            st.subheader("1. 场景类型分布")
            if "type" in df_cases.columns:
                type_counts = df_cases["type"].value_counts().reset_index()
                type_counts.columns = ["类型", "数量"]
                fig_pie = px.pie(type_counts, values="数量", names="类型", hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("用例中未包含 type 字段，无法统计场景类型分布。")

            # 2. 结构评测
            metrics = compute_basic_metrics(cases)
            st.subheader("2. 结构 & 描述质量（规则评测）")
            c1, c2, c3 = st.columns(3)
            c1.metric("格式合规率", f"{metrics['format_rate'] * 100:.1f}%")
            c2.metric("冗余度", f"{metrics['redundancy'] * 100:.1f}%", help="越高说明标题重复越多（按模块+标题统计）")
            rigor_score = max(100 - metrics["vague_count"] * 10, 0)
            c3.metric("描述严谨度", f"{rigor_score:.1f} / 100")

            if metrics["vague_count"] > 0:
                st.warning(
                    f"检测到 {metrics['vague_count']} 处模糊词（如“等等/大概/可能”等），"
                    "建议优化 Prompt 或通过自我修正功能改写用例。"
                )
            else:
                st.success("未检测到明显模糊词，用例描述较为严谨。")

            # 3. 统计+相似度雷达图
            if run_eval:
                st.markdown("---")
                st.subheader("3. 综合雷达图评估")

                # 文本 Jaccard（作为一个很粗糙的对照）
                text_sim = 0.0
                if golden_text.strip():
                    ai_concat = "".join(df_cases["title"].astype(str).tolist())
                    text_sim = jaccard_similarity(ai_concat, golden_text) * 100

                # CSV/Excel 相似度（Jaccard + 语义 + F1）
                csv_jac = csv_sem = csv_rec = csv_pre = csv_f1 = None
                if uploaded_gold is not None:
                    try:
                        if uploaded_gold.name.endswith(".csv"):
                            human_df = pd.read_csv(uploaded_gold)
                        else:
                            human_df = pd.read_excel(uploaded_gold)
                        sim_dict = evaluate_against_human_csv(cases, human_df)
                        csv_jac = sim_dict["jaccard"]
                        csv_sem = sim_dict["semantic"]
                        csv_rec = sim_dict["recall"]
                        csv_pre = sim_dict["precision"]
                        csv_f1 = sim_dict["f1"]
                        st.info(
                            f"基于 CSV/Excel 的标题相似度："
                            f"Jaccard ≈ {csv_jac:.1f}%，"
                            f"语义相似度 ≈ {csv_sem:.1f}%，"
                            f"召回率 ≈ {csv_rec:.1f}%，"
                            f"精确率 ≈ {csv_pre:.1f}%，"
                            f"F1 ≈ {csv_f1:.1f}%。"
                        )
                    except Exception as e:
                        st.error(f"解析人工用例文件失败：{e}")

                # 类型丰富度评分
                if "type" in df_cases.columns:
                    types_set = set(df_cases["type"].dropna().tolist())
                else:
                    types_set = set()

                has_positive = "正向" in types_set
                has_negative = "异常" in types_set

                if has_positive and has_negative:
                    base_score = 85.0
                elif has_positive or has_negative:
                    base_score = 70.0
                else:
                    base_score = 60.0

                extra_types = {"边界", "安全", "性能", "界面"}
                extra_count = len(types_set & extra_types)
                extra_bonus = min(15.0, extra_count * 5.0)

                balance_score = min(100.0, base_score + extra_bonus)

                format_score = metrics["format_rate"] * 100
                redundancy_score = (1 - metrics["redundancy"]) * 100
                rigor_score_final = rigor_score
                sim_score = csv_f1 if (csv_f1 is not None) else (text_sim if golden_text.strip() else 0.0)

                categories = ["场景类型丰富度", "格式规范性", "冗余控制", "描述严谨度", "对人工基准的接近度(F1)"]
                scores = [
                    balance_score,
                    format_score,
                    redundancy_score,
                    rigor_score_final,
                    sim_score,
                ]

                fig_radar = go.Figure()
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=scores,
                        theta=categories,
                        fill="toself",
                        name="当前版本",
                    )
                )
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=False,
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            # 4. LLM 评审
            if run_judge:
                if not ark_api_key:
                    st.error("请先在侧边栏配置 API Key")
                elif not prd_text.strip():
                    st.error("当前 PRD 为空，无法进行 LLM 评审")
                else:
                    with st.spinner("正在调用评审模型，请稍候..."):
                        try:
                            judge = judge_by_llm(ark_api_key, judge_model_id, prd_text, cases)
                            st.session_state["judge_result"] = judge
                            st.success("LLM 评审完成")
                        except Exception as e:
                            st.error(f"评审失败：{e}")

            judge = st.session_state["judge_result"]
            if judge:
                st.markdown("---")
                st.subheader("4. LLM 评分结果（近似需求覆盖率 & 清晰度）")
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("完整性/需求覆盖率", f"{judge.get('completeness_score', 0):.1f} / 10")
                cc2.metric("清晰度评分", f"{judge.get('clarity_score', 0):.1f} / 10")
                cc3.metric("综合评分", f"{judge.get('overall_score', 0):.1f} / 10")
                st.markdown("**LLM 评审点评：**")
                st.write(judge.get("comments", "（模型未给出详细点评）"))

            # 5. 功能点覆盖检查
            if run_coverage:
                if not ark_api_key:
                    st.error("请先在侧边栏配置 API Key")
                elif not prd_text.strip():
                    st.error("当前 PRD 为空，无法进行覆盖率检查")
                elif not features:
                    st.error("当前没有功能点列表，建议使用精细模式生成后再进行覆盖率检查。")
                else:
                    with st.spinner("正在进行功能点覆盖率分析..."):
                        try:
                            cov = coverage_by_llm(ark_api_key, judge_model_id, prd_text, features, cases)
                            st.session_state["coverage_result"] = cov
                            st.success("覆盖率分析完成")
                        except Exception as e:
                            st.error(f"覆盖率分析失败：{e}")

            coverage_result = st.session_state["coverage_result"]
            if coverage_result:
                st.markdown("---")
                st.subheader("5. 功能点覆盖率（LLM 检查版本）")
                cov_score = coverage_result.get("coverage_score", 0) * 100 if coverage_result.get("coverage_score", 0) <= 1.0 else coverage_result.get("coverage_score", 0)
                st.metric("功能点覆盖率", f"{cov_score:.1f}%")
                uncovered = coverage_result.get("uncovered_features", [])
                if uncovered:
                    st.warning(f"存在 {len(uncovered)} 个功能点未被任何用例覆盖：{', '.join(uncovered)}")
                st.markdown("**LLM 分析说明：**")
                st.write(coverage_result.get("analysis", "（模型未给出分析）"))

            # 6. 幻觉检测
            if run_hallu:
                if not ark_api_key:
                    st.error("请先在侧边栏配置 API Key")
                elif not prd_text.strip():
                    st.error("当前 PRD 为空，无法进行幻觉检测")
                else:
                    with st.spinner("正在对部分用例进行幻觉检测..."):
                        try:
                            hallu = hallucination_check_by_llm(ark_api_key, judge_model_id, prd_text, cases)
                            st.session_state["hallucination_result"] = hallu
                            st.success("幻觉检测完成")
                        except Exception as e:
                            st.error(f"幻觉检测失败：{e}")

            hallu = st.session_state["hallucination_result"]
            if hallu:
                st.markdown("---")
                st.subheader("6. 幻觉检测结果")
                suspicious = hallu.get("suspicious_cases", [])
                if suspicious:
                    st.warning(f"检测到 {len(suspicious)} 条疑似幻觉用例（预期结果在 PRD 中缺乏依据）：")
                    for item in suspicious:
                        st.write(f"- {item.get('id','(未知ID)')}: {item.get('reason','未给出原因')}")
                else:
                    st.success("未检测到明显的幻觉用例。")
                st.markdown("**LLM 总结：**")
                st.write(hallu.get("summary", "（模型未给出总结）"))

            # 7. Self-Correction：根据评审意见自动优化
            if run_improve:
                if not ark_api_key:
                    st.error("请先在侧边栏配置 API Key")
                elif not prd_text.strip():
                    st.error("当前 PRD 为空，无法进行自我修正")
                else:
                    with st.spinner("正在根据评审意见自动优化用例..."):
                        try:
                            new_cases = improve_cases_with_llm(
                                api_key=ark_api_key,
                                model_id=judge_model_id,
                                prd_text=prd_text,
                                guidelines=test_guidelines,
                                cases=cases,
                                judge_result=st.session_state.get("judge_result"),
                            )
                            st.session_state["cases"] = new_cases
                            st.success(f"已生成优化后的 {len(new_cases)} 条用例，请返回 Tab2 查看最新结果。")
                        except Exception as e:
                            st.error(f"自我修正失败：{e}")
