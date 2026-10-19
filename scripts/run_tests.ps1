# Run the CrowdSense AI test suite with native thread pools pinned to 1.
# Mixed OpenMP runtimes (torch / OpenBLAS / OpenCV) cause intermittent
# access-violation crashes on Windows when they spawn competing thread pools.
# Setting the variables in the shell (before Python starts) is the reliable fix;
# tests/conftest.py pins them too as belt-and-braces.

$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:OMP_DYNAMIC = "FALSE"
$env:MKL_DYNAMIC = "FALSE"
# Tolerate torch (libiomp) + OpenBLAS/OpenCV OpenMP coexistence on Windows.
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
# Hide the GPU: belt-and-braces. The CUDA xgboost wheel was the original culprit
# (nvdxgdmal64.dll unload race -> 0xC0000005); requirements.txt now pins
# xgboost-cpu, but keeping -1 guards any other CUDA probing. CPU-only project.
$env:CUDA_VISIBLE_DEVICES = "-1"
# Same NVIDIA-driver race via OpenCV OpenCL; force the CPU path.
$env:OPENCV_OPENCL_RUNTIME = "disabled"

python -m pytest tests/ @args
exit $LASTEXITCODE
