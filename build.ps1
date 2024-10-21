# A very easy way to build the application

# Check if CL.exe exist
if (!(Get-Command "cl.exe" -ErrorAction SilentlyContinue)) {
    Write-Host "cl.exe not found. Please install Visual Studio 2022 or run in Developer Prompt" -ForegroundColor Red
    exit
}

# The cuDNN path
$env:CUDNN_LIB="C:\Program Files\NVIDIA GPU Computing Toolkit\cuDNN\v9.2-12"
# The CUDA path
$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
# Additional PATH for CUDA binary/toolkit (although you can also set it in PATH yourself)
# I did this since I have CUDA 11.8 that I'm using for chaiNNer
$env:PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\libnvvp;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;$env:PATH"

# Build with CUDA
# Pass other arguments from the command line
cargo build --release --locked --features cuda-compute $args
