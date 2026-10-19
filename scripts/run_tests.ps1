# Run the CrowdSense AI test suite with native thread pools pinned to 1.
# Mixed OpenMP runtimes (torch / OpenBLAS / OpenCV) cause intermittent
# access-violation crashes on Windows when they spawn competing thread pools.
# Setting the variables in the shell (before Python starts) is the reliable fix;
# tests/conftest.py pins them too as belt-and-braces.

$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

python -m pytest tests/ @args
exit $LASTEXITCODE
