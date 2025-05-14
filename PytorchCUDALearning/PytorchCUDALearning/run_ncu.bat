@echo off
set /p PYTHON_PATH=<python_path.txt
echo Using Python at: %PYTHON_PATH%

"C:\Program Files\NVIDIA Corporation\Nsight Compute 2023.1.1\target\windows-desktop-x64\ncu.exe" --export my_profile_report --force-overwrite ^
    --kernel-regex "add_arrays_kernel" ^
    "%PYTHON_PATH%" build_and_test.py

echo Profile completed.