"""
角色名工具 - 标准化、模糊匹配，无重依赖
"""

import re
from typing import Dict, Set, Optional


def normalize_character_with_mapping(
    name: str,
    mapping: Dict[str, str],
    threshold: float = 0.75,
) -> str:
    """
    使用映射表标准化角色名，支持模糊匹配

    Args:
        name: 原始角色名
        mapping: 别名 -> 规范名 映射
        threshold: 编辑距离相似度阈值 (0~1)
    """
    name = (name or "").strip().upper()
    if not name:
        return ""
    if name in mapping:
        return mapping[name]
    for alias, canonical in mapping.items():
        if alias.upper() == name:
            return canonical
    if threshold < 1.0 and mapping:
        best = fuzzy_match_character(name, list(mapping.keys()) + list(set(mapping.values())), threshold)
        if best:
            return mapping.get(best, best)
    return name


def fuzzy_match_character(
    name: str,
    candidates: list,
    threshold: float = 0.75,
) -> Optional[str]:
    """
    模糊匹配角色名（基于子串与编辑距离）

    Args:
        name: 待匹配名
        candidates: 候选名列表
        threshold: 相似度阈值
    """
    name = (name or "").strip().upper()
    if not name or not candidates:
        return None
    name_parts = set(name.split())
    best_match, best_score = None, 0.0
    for c in candidates:
        c_upper = (c or "").strip().upper()
        if not c_upper:
            continue
        if name == c_upper:
            return c_upper
        c_parts = set(c_upper.split())
        overlap = len(name_parts & c_parts) / max(len(name_parts), 1)
        if overlap >= threshold and overlap > best_score:
            best_match, best_score = c_upper, overlap
    return best_match


def normalize_character_string(
    chars_str: str,
    known_chars: Optional[Set[str]] = None,
) -> str:
    """
    标准化角色字符串，合并重复、映射到已知角色

    Args:
        chars_str: 如 "SARA, A WOMAN, GRAHAM"
        known_chars: 已知规范角色名集合
    """
    if not chars_str:
        return ""
    known = known_chars or set()
    parts = re.split(r"[,;]", chars_str)
    seen = set()
    result = []
    for p in parts:
        p = p.strip().upper()
        if not p or len(p) < 2:
            continue
        if p in ("NONE", "N/A", "-", "A", "THE"):
            continue
        if p in known:
            if p not in seen:
                seen.add(p)
                result.append(p)
            continue
        matched = fuzzy_match_character(p, list(known), 0.6) if known else None
        if matched and matched not in seen:
            seen.add(matched)
            result.append(matched)
        elif not matched and p not in seen:
            seen.add(p)
            result.append(p)
    return ", ".join(result)
