#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
急救伤情理解 HTTP API 服务（供其他模型/系统调用）

接口：
  GET  /health      健康检查，返回 {"status":"ok"}
  POST /classify    伤情分类，请求体 {"text":"报案原文"}，返回结构化 JSON

用法：
  python3 api_server.py                 # 默认监听 0.0.0.0:8080
  python3 api_server.py --port 9000     # 自定义端口

另一个模型调用示例（Python）：
  import requests
  r = requests.post("http://<pi_ip>:8080/classify",
                    json={"text": "被刀割伤直流血"})
  print(r.json())

返回示例：
{
  "original": "被刀割伤直流血",
  "injuries": [{
    "idx": 1,
    "injury": "被刀割伤直流血",
    "type": "开放式",
    "group": "机械性损伤",
    "detail": "切割伤",
    "location": "躯干",
    ...
  }]
}
"""

import os
import sys
import json
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from pi_inference import build_classifier

# 全局分类器（进程启动时加载，避免每次请求重载模型）
_classify_fn = None


def _get_model_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.isdir(os.path.join(here, "models")):
        return os.path.join(here, "models")
    return os.path.join(os.path.dirname(os.path.dirname(here)), "models", "gguf")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # 简化日志
        sys.stderr.write("[api] %s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/classify":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._send_json(400, {"error": "empty body"})
            return
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            text = data.get("text", "")
            if not text:
                self._send_json(400, {"error": "missing 'text'"})
                return
        except Exception as e:
            self._send_json(400, {"error": "invalid json: %s" % e})
            return

        try:
            # 单轮分类（不追问，直接输出结果；若要追问可用交互式 pi_inference.py）
            result = _classify_fn(text, ask_fn=None)
            self._send_json(200, result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._send_json(500, {"error": str(e)})


def main():
    global _classify_fn
    ap = argparse.ArgumentParser(description="急救伤情理解 HTTP API")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model-dir", type=str, default=None)
    args = ap.parse_args()

    model_dir = args.model_dir or _get_model_dir()
    print(f"[api] 加载模型目录: {model_dir}")
    _classify_fn, _ = build_classifier(model_dir)
    print("[api] 模型加载完成")

    server = HTTPServer((args.host, args.port), Handler)
    print(f"[api] 监听 http://{args.host}:{args.port}")
    print(f"[api] 接口: POST /classify  body={{\"text\":\"...\"}}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[api] 退出")
        server.server_close()


if __name__ == "__main__":
    main()
