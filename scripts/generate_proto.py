from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    proto_dir = root / "proto"
    proto_file = proto_dir / "rag.proto"
    out_dir = root / "src" / "pitchavatar_rag_sentinel" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{proto_dir}",
        f"--python_out={out_dir}",
        f"--pyi_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        str(proto_file),
    ]

    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return result.returncode

    grpc_file = out_dir / "rag_pb2_grpc.py"
    content = grpc_file.read_text(encoding="utf-8")
    content = content.replace("import rag_pb2 as rag__pb2", "from . import rag_pb2 as rag__pb2")
    grpc_file.write_text(content, encoding="utf-8")

    init_file = out_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("", encoding="utf-8")

    print(f"Generated protobuf stubs into {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

