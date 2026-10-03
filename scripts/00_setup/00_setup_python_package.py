import subprocess
from pathlib import Path
import sys


def setup_python_package():
	print("=" * 60)
	print(" 🛠️  Setting up Python package...")
	print("=" * 60)

	root_dir = Path(__file__).resolve().parents[2]
	requirements_path = root_dir / "requirements.txt"

	if not (root_dir / "pyproject.toml").exists():
		print(f"❌ Error: pyproject.toml not found at {root_dir}")
		print("   Please check the directory structure.")
		sys.exit(1)

	if not requirements_path.is_file():
		print(f"❌ Error: requirements.txt not found at {requirements_path}")
		sys.exit(1)

	try:
		print(f"Installing dependencies from {requirements_path.name}...")
		subprocess.run(
			[sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
			check=True,
			cwd=root_dir,
		)
		print(" ✅ Dependencies installed successfully!\n")

		subprocess.run(
			[sys.executable, "-m", "pip", "install", "-e", "."],
			check=True,
			cwd=root_dir,
		)
		print(" ✅ Python package installed successfully!\n")

	except subprocess.CalledProcessError as e:
		print(f"❌ Failed to install Python package. Error: {e}")
		sys.exit(1)


if __name__ == "__main__":
	setup_python_package()
