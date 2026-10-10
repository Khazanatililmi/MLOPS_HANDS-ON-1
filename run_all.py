import subprocess
import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

env = os.environ.copy()
env["PYTHONUTF8"] = "1"
env["PYTHONIOENCODING"] = "utf-8"

scripts = [
    "01_basic_logging.py",
    "02_pytorch_logging.py",
    "03_autologging.py",
    "04_optuna_nested_runs.py",
    "05_search_and_compare.py",
    "06_model_registry.py"
]

print("=== STARTING HANDS-ON 1 EXECUTION ===")
for script in scripts:
    print(f"\n---> Running {script}...")
    res = subprocess.run([sys.executable, script], capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")
    out_clean = res.stdout.encode('ascii', errors='ignore').decode('ascii')
    err_clean = res.stderr.encode('ascii', errors='ignore').decode('ascii')
    print(out_clean)
    if err_clean:
        print("STDERR:", err_clean)
    if res.returncode != 0:
        print(f"Error executing {script}, exiting with code {res.returncode}.")
        sys.exit(res.returncode)

print("\n=== ALL HANDS-ON 1 SCRIPTS EXECUTED SUCCESSFULLY! ===")
