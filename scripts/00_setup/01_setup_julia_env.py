import subprocess
import sys
from pathlib import Path

from src.paths import PROJECT_ROOT, JULIA_PROJECT_DIR

########################################
# 3. ビルドするときにDelaunayTriangulationが重くて動かないがち.
# sudo fallocate -l 8G /swapfile_large
# sudo chmod 600 /swapfile_large
# sudo mkswap /swapfile_large
# sudo swapon /swapfile_large
# を実行してスワップ領域を確保してから実行する必要あり. 
########################################

def build_sysimage():
    print("="*60)
    print(" 🛠️  Building Julia Sysimage for Simulation...")
    print("    This process may take a few minutes. Please wait.")
    print(" ⚠️  Note: This process is memory-intensive (approx 9GB). If the build fails, please ensure you have sufficient RAM or increase your swap space.")
    print("="*60)

    build_script_path = JULIA_PROJECT_DIR / "build_tools" / "make_sysimage.jl"

    if not build_script_path.exists():
        print(f"Error: Build script not found at {build_script_path}")
        sys.exit(1)

    cmd = [
        "julia",
        f"--project={JULIA_PROJECT_DIR}",
        str(build_script_path)
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=PROJECT_ROOT
        )

    # ログをリアルタイム表示
        for line in process.stdout:
            print(line, end="")
        
        process.wait()

        if process.returncode == 0:
            print("\n" + "="*60)
            print(" ✅ Sysimage created successfully!")
            print(f"    Location: {JULIA_PROJECT_DIR / 'build_tools' / 'sys_simulation.so'}")
            print("="*60)
        else:
            print("\n❌ Build failed with return code:", process.returncode)
            sys.exit(process.returncode)

    except FileNotFoundError:
        print("Error: 'julia' command not found. Please install Julia and add it to PATH.")
        sys.exit(1)

if __name__ == "__main__":
    build_sysimage()