@echo off
REM ==========================================
REM Setup Python venv and install packages (pip-only)
REM ==========================================

REM Step 1: Create virtual environment
python -m venv venv_cv

REM Step 2: Activate venv
call venv_cv\Scripts\activate

REM Step 3: Upgrade pip
pip install --upgrade pip

REM Step 4: Install PyTorch CPU first (pins numpy correctly)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

REM Step 5: Install remaining packages from requirements.txt
pip install -r requirements.txt -c constraints.txt

pip install -U openmim
mim install mmengine

mim install mmcv-lite

REM Step 6: Verify installations
python -c "import torch, torchvision, numpy as np, cv2; print('torch:', torch.__version__, 'torchvision:', torchvision.__version__, 'numpy:', np.__version__)"

// cd src\external\mmpose
pip install -r requirements.txt
pip install -e .
echo.
echo Setup complete! Your venv_cv environment is ready.
pause