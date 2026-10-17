"""pytest 全局配置：把 src/ 加入导入路径。

这样测试里可以直接 `from metrics import ...`，与脚本内的导入方式一致。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
