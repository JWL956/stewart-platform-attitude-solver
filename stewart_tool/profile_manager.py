# -*- coding: utf-8 -*-
"""档案管理器：命名建档、选择、编辑、删除、自动持久化。

档案以 JSON 文件保存在 profiles/ 目录，每个档案一个文件。
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def app_data_dir() -> Path:
    """应用数据根目录。

    打包为 exe 后 __file__ 指向临时解压目录，若以其为基准会导致档案
    「每次关闭即丢失」。因此 frozen 模式统一落到 exe 所在目录。
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


PROFILES_DIR = app_data_dir() / "profiles"


def safe_filename(name: str) -> str:
    """将档案名转为安全文件名（保留中文，去除非法字符）。"""
    cleaned = re.sub(r'[\\/:*?"<>|\r\n\t]', "_", name).strip(" .")
    return (cleaned or "未命名档案")[:80] + ".json"


class ProfileManager:
    def __init__(self, directory: Path = PROFILES_DIR):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[dict]:
        """返回档案摘要列表，按修改时间倒序。"""
        items = []
        for f in self.directory.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                items.append({
                    "file": str(f),
                    "name": data.get("name", f.stem),
                    "structure": data.get("config", {}).get("structure", "SPU"),
                    "created": data.get("created", ""),
                    "modified": data.get("modified", ""),
                    "note": data.get("note", ""),
                })
            except Exception:
                continue
        items.sort(key=lambda x: x.get("modified", ""), reverse=True)
        return items

    def exists(self, name: str) -> bool:
        return (self.directory / safe_filename(name)).exists()

    def load(self, name: str) -> Optional[dict]:
        """按名称加载档案完整数据。"""
        path = self.directory / safe_filename(name)
        if not path.exists():
            # 兼容：遍历查找 name 字段匹配的档案
            for f in self.directory.glob("*.json"):
                try:
                    d = json.loads(f.read_text(encoding="utf-8"))
                    if d.get("name") == name:
                        path = f
                        break
                except Exception:
                    continue
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def save(self, name: str, config: dict, note: str = "", fk_inputs: list = None) -> str:
        """保存/更新档案，返回档案文件路径。"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        existing = self.load(name)
        data = {
            "name": name,
            "structure": config.get("structure", "SPU"),
            "created": existing.get("created", now) if existing else now,
            "modified": now,
            "note": note,
            "config": config,
            "fk_inputs": fk_inputs or (existing.get("fk_inputs", []) if existing else []),
        }
        path = self.directory / safe_filename(name)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def delete(self, name: str) -> bool:
        path = self.directory / safe_filename(name)
        if path.exists():
            path.unlink()
            return True
        for f in self.directory.glob("*.json"):
            try:
                if json.loads(f.read_text(encoding="utf-8")).get("name") == name:
                    f.unlink()
                    return True
            except Exception:
                continue
        return False

    def export_package(self, name: str, path) -> bool:
        """把档案导出为可分享的包（.json，含完整配置与输入）。"""
        data = self.load(name)
        if data is None:
            return False
        data["_package"] = "stewart-profile/1.0"
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return True

    def import_package(self, path) -> Optional[str]:
        """从包文件导入档案，返回保存后的档案名；格式无效返回 None。"""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(data, dict) or "config" not in data:
            return None
        name = str(data.get("name") or Path(path).stem)
        if not name.strip():
            name = Path(path).stem
        # 避免重名覆盖：自动追加后缀
        base = name; i = 1
        while self.exists(name):
            name = f"{base}_导入{i}"; i += 1
        data["name"] = name
        data.pop("_package", None)
        data["modified"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        (self.directory / safe_filename(name)).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return name

    def rename(self, old_name: str, new_name: str) -> bool:
        data = self.load(old_name)
        if data is None:
            return False
        if self.exists(new_name):
            return False
        data["name"] = new_name
        data["modified"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        (self.directory / safe_filename(new_name)).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.delete(old_name)
        return True

    def backup(self, name: str) -> Optional[str]:
        """创建档案备份（.bak）。"""
        path = self.directory / safe_filename(name)
        if not path.exists():
            return None
        bak = path.with_suffix(".json.bak")
        shutil.copy2(path, bak)
        return str(bak)
