import subprocess
import sys


def test_mate_models_imports_in_fresh_process():
    subprocess.run([sys.executable, '-c', 'import mate.models'], check=True)
