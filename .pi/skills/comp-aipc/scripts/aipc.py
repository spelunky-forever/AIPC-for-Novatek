#!/usr/bin/env python3
"""
aipc.py - Unified CLI Entrypoint for Qualcomm QAIRT Workflow (ARM Linux & x86 Linux)
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_SETUP = "/home/spelunky-forever/workplace/AIPC-for-Novatek/env_setup.sh"


def _is_arm64() -> bool:
    return platform.machine().lower() in ("aarch64", "arm64")


def _host_arch_dir() -> str:
    return "aarch64-oe-linux-gcc11.2" if _is_arm64() else "x86_64-linux-clang"


def _ensure_env() -> str:
    sdk = os.environ.get("QAIRT_SDK_ROOT", "")
    if not sdk:
        print(f"[ERROR] QAIRT_SDK_ROOT is not set. Run:\n  source {DEFAULT_ENV_SETUP}")
        sys.exit(1)
    return sdk


def cmd_init(args):
    cmd = [sys.executable, str(SCRIPT_DIR / "qai_project_setup.py"), args.project_dir]
    sys.exit(subprocess.run(cmd).returncode)


def cmd_preflight(args):
    sdk = _ensure_env()
    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.exists():
        print(f"[FAIL] ONNX not found: {onnx_path}")
        sys.exit(1)

    tmp_dir = os.environ.get("QAIRT_TMP_DIR", "/tmp")
    tmp_free_gb = shutil.disk_usage(tmp_dir).free / (1024**3)
    work_free_gb = shutil.disk_usage(".").free / (1024**3)
    print(f"[OK] Host Arch      = {platform.machine()} ({_host_arch_dir()})")
    print(f"[OK] Python Version = {sys.version.split()[0]}")
    print(f"[OK] QAIRT_SDK_ROOT = {sdk}")
    print(f"[INFO] Temp ({tmp_dir}) free: {tmp_free_gb:.1f} GB | Workdir free: {work_free_gb:.1f} GB")
    if tmp_free_gb < 2.0:
        print("[WARN] Low /tmp space! Set 'export QAIRT_TMP_DIR=\"$PWD/qairt_tmp\"' for large models.")

    import onnx
    onnx.checker.check_model(str(onnx_path))
    m = onnx.load(str(onnx_path), load_external_data=False)
    inputs = [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]
    outputs = [o.name for o in m.graph.output]
    print(f"[OK] ONNX valid: {onnx_path.name}")
    print(f"     Inputs : {inputs}")
    print(f"     Outputs: {outputs}")


def cmd_inspect(args):
    sdk = _ensure_env()
    onnx_path = str(Path(args.onnx).resolve())
    subprocess.run([sys.executable, str(SCRIPT_DIR / "qai_inspect_onnxio.py"), onnx_path], check=True)
    converter = Path(sdk) / "bin" / _host_arch_dir() / "qnn-onnx-converter"
    if converter.exists():
        print("\n[INFO] Running qnn-onnx-converter --dry_run ...")
        subprocess.run([str(converter), "--input_network", onnx_path, "--dry_run"])


def cmd_convert(args, extra):
    _ensure_env()
    target_script = "qai_convert_fp.py" if args.so_flow else "run_pipeline.py"
    cmd = [sys.executable, str(SCRIPT_DIR / target_script)] + extra
    sys.exit(subprocess.run(cmd).returncode)


def cmd_ctxbin(args, extra):
    _ensure_env()
    ctx_script = "qai_dev_gen_contextbin.py" if _is_arm64() else "qai_dev_gen_contextbin_x86.py"
    cmd = [sys.executable, str(SCRIPT_DIR / ctx_script)] + extra
    sys.exit(subprocess.run(cmd).returncode)


def cmd_infer(args, extra):
    cmd = [sys.executable, str(SCRIPT_DIR / "qai"), args.script] + extra
    sys.exit(subprocess.run(cmd).returncode)


def main():
    p = argparse.ArgumentParser(prog="aipc.py", description="Unified AIPC CLI for Qualcomm QAIRT")
    sub = p.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Bootstrap a new AIPC project directory")
    p_init.add_argument("project_dir")
    p_init.set_defaults(func=cmd_init)

    p_pre = sub.add_parser("preflight", help="Check SDK, Python version, /tmp space, and ONNX (>2GB safe)")
    p_pre.add_argument("onnx")
    p_pre.set_defaults(func=cmd_preflight)

    p_ins = sub.add_parser("inspect", help="Generate <model>.yaml and run converter --dry_run")
    p_ins.add_argument("onnx")
    p_ins.set_defaults(func=cmd_inspect)

    p_conv = sub.add_parser("convert", help="Run conversion/quantization (forwards flags to run_pipeline.py)")
    p_conv.add_argument("--so-flow", action="store_true", help="Use .cpp/.so flow (qai_convert_fp.py) instead of DLC flow")
    p_conv.set_defaults(func=cmd_convert)

    p_ctx = sub.add_parser("ctxbin", help="Generate context binary (uses native generator on ARM64)")
    p_ctx.set_defaults(func=cmd_ctxbin)

    p_inf = sub.add_parser("infer", help="Run inference script through qai wrapper")
    p_inf.add_argument("script")
    p_inf.set_defaults(func=cmd_infer)

    args, extra = p.parse_known_args()
    if args.command in ("convert", "ctxbin", "infer"):
        args.func(args, extra)
    else:
        args.func(args)


if __name__ == "__main__":
    main()