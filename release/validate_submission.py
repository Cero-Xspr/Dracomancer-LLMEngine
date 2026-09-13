#!/usr/bin/env python3
"""校验一份社区提交的模型适配档案（issue 里的 JSON）。

★ 与 draco.py 共用同一份 schema 校验（draco._validate_profile）——
  两处实现必然漂移，一处实现才可信。

用法：
  python3 release/validate_submission.py <提交.json> [--gguf <本地模型.gguf>]
  python3 release/validate_submission.py --from-issue-body <issue正文.md>

退出码：0=通过，1=拒绝。CI 里拿退出码打 label；本地拿它当提交前自检。
"""
import sys, os, json, argparse

# ★ 只依赖 adapt_schema（与 draco 共用同一份 schema 实现）—— 本文件可原样拷进公开
#   仓库独立运行，不需要引擎在场。仓库里它俩放在同一目录（release/）。
for _p in (os.path.dirname(os.path.abspath(__file__)),
           os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import adapt_schema  # noqa: E402


def check(p, gguf_path=None):
    errs, warns = [], []
    try:
        adapt_schema.validate_profile(p, "<submission>")
    except Exception as e:
        return [f"schema: {e}"], warns
    src = p.get("source") or {}
    if not src.get("author"):
        errs.append("source.author 缺失（GitHub ID）")
    if not (p.get("notes") or "").strip():
        warns.append("notes 为空——踩坑记录是档案最有价值的部分，强烈建议填写")
    if p.get("requires_local_build"):
        warns.append("requires_local_build=true 只能由维护者确认后保留；提交会被人工复核")
    if gguf_path:
        meta = adapt_schema.gguf_meta(gguf_path, want=("general.architecture", "general.name"))
        arch = meta.get("general.architecture")
        if arch != (p.get("match") or {}).get("arch"):
            errs.append(f"match.arch={p.get('match', {}).get('arch')!r} 与 GGUF 实际架构 {arch!r} 不符")
        sub = (p.get("match") or {}).get("name_contains")
        if sub and sub.lower() not in (meta.get("general.name") or "").lower():
            errs.append(f"name_contains={sub!r} 未命中 GGUF 的 general.name={meta.get('general.name')!r}")
    return errs, warns


def from_issue_body(md):
    """从 issue 正文提取 ```json 围栏块，返回候选列表。"""
    out, cur = [], []
    inside = False
    for ln in md.splitlines():
        if not inside and ln.strip().startswith("```json"):
            inside, cur = True, []
        elif inside and ln.strip() == "```":
            inside = False
            try:
                out.append(json.loads("\n".join(cur)))
            except Exception:
                pass
        elif inside:
            cur.append(ln)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="提交的 JSON 文件，或（配 --from-issue-body）issue 正文")
    ap.add_argument("--gguf", help="可选：本地 GGUF 路径，校验 match 规则是否真的命中")
    ap.add_argument("--from-issue-body", action="store_true", dest="from_body",
                    help="把 path 当 issue 正文，提取其中的 ```json 块逐一校验")
    a = ap.parse_args()

    if a.from_body:
        cands = from_issue_body(open(a.path, encoding="utf-8", errors="replace").read())
        if not cands:
            print("FAIL: issue 正文里没有可解析的 ```json 块")
            return 1
        rc = 1
        for i, p in enumerate(cands):
            if not isinstance(p, dict):
                continue
            errs, warns = check(p, a.gguf)
            tag = "PASS" if not errs else "FAIL"
            if not errs:
                rc = 0
            print(f"--- 候选 #{i}: {tag}")
            for e in errs:
                print(f"  [err]  {e}")
            for w in warns:
                print(f"  [warn] {w}")
        return rc

    try:
        p = json.load(open(a.path, encoding="utf-8"))
    except Exception as e:
        print(f"FAIL: 不是合法 JSON：{e}")
        return 1
    errs, warns = check(p, a.gguf)
    for e in errs:
        print(f"[err]  {e}")
    for w in warns:
        print(f"[warn] {w}")
    print(("PASS" if not errs else "FAIL") + f": {p.get('match', {}).get('arch', '?')}")
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main())
