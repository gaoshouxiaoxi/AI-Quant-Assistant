"""用 Playwright 驱动本机已安装的 Edge 给 Streamlit 界面截图。

不需要下载 Chromium：Playwright 支持 `channel="msedge"` 直接复用系统 Edge。

用法：
    1. 先启动服务：streamlit run app.py --server.port 8501
    2. 再执行：python scripts/take_screenshots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "screenshots"
URL = "http://127.0.0.1:8501"

# (标签名, 输出文件名, 需要等待出现的文本)
PAGES = [
    ("概览", "streamlit-overview.png", "区间收益"),
    ("均线策略", "streamlit-strategy.png", "最终资产"),
    ("波动率预测", "streamlit-volatility.png", "未来 5 日预测"),
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1680, "height": 1050},
                                device_scale_factor=1)
        page.goto(URL, wait_until="domcontentloaded", timeout=60_000)

        for tab_label, filename, wait_text in PAGES:
            try:
                page.get_by_role("tab", name=tab_label).click(timeout=30_000)
            except Exception as exc:  # noqa: BLE001
                print(f"[{tab_label}] 点击标签失败：{type(exc).__name__}: {exc}")
                continue

            try:
                page.get_by_text(wait_text, exact=False).first.wait_for(
                    state="visible", timeout=90_000)
            except Exception as exc:  # noqa: BLE001
                print(f"[{tab_label}] 等待内容超时：{type(exc).__name__}")
                continue

            # 关键：st.pyplot 渲染的是图片，必须等图片真正加载完再截图，
            # 否则会拍到"有指标、无图表"的半成品页面。
            try:
                img = page.locator("[data-testid='stImage'] img").first
                img.wait_for(state="visible", timeout=90_000)
                page.wait_for_function(
                    "() => { const im = document.querySelector("
                    "\"[data-testid='stImage'] img\");"
                    " return im && im.complete && im.naturalWidth > 0; }",
                    timeout=60_000,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[{tab_label}] 等待图表未成功：{type(exc).__name__}")

            # 图表加载后可能还有布局重排，再稳一下
            page.wait_for_timeout(3000)

            target = OUT_DIR / filename
            page.screenshot(path=str(target), full_page=True)
            size_kb = target.stat().st_size / 1024
            print(f"[{tab_label}] 已保存 {filename}（{size_kb:.0f} KB）")

        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
