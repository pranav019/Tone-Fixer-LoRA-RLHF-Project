# Run with: python src/verify_local.py

import sys
import importlib

print("=" * 55)
print("  LOCAL ENVIRONMENT VERIFICATION — tone-fixer (Conda)")
print("=" * 55)

# Python version
print(f"\n✅ Python : {sys.version}")

# Check packages
packages = [
    "torch", "transformers", "datasets",
    "peft", "accelerate", "pandas", "numpy"
]

all_good = True
for pkg in packages:
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, "__version__", "installed")
        print(f"  ✅ {pkg:<15}: {version}")
    except ImportError:
        print(f"  ❌ {pkg:<15}: NOT FOUND — run: pip install {pkg}")
        all_good = False

# PyTorch specific check
import torch
print(f"\n📦 PyTorch Details:")
print(f"   Version  : {torch.__version__}")
print(f"   CUDA     : {torch.cuda.is_available()}")
print(f"   Device   : {'CUDA' if torch.cuda.is_available() else 'CPU'}")

# For your laptop, CUDA should be False — that's correct
if not torch.cuda.is_available():
    print(f"   ✅ CPU-only mode confirmed (correct for Lenovo S540)")

if all_good:
    print("\n🎉 Local Conda environment is fully ready!")
    print("   Next step: Set up Google Colab")
else:
    print("\n⚠️  Fix the missing packages above, then re-run.")