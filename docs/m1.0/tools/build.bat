@echo off
rem Marian M1.0: llama.cpp a37cc46104eb50900d5ab4a9ba02a0abdac039db + P0 trace patch (branch marian/m1.0),
rem CUDA for sm_89 (RTX 4090 Laptop), on the reference laptop. Toolchain and network-off switches as <baseline>\build.bat.
setlocal
set "R=<research>"
set "SRC=%R%\llama.cpp"
set "BUILD=%R%\build"
set "CUDAROOT=<maya-build>\env\Library"
set "CMAKE=<vs-buildtools>\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA=<vs-buildtools>\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
set "JOBS=%~1"
if "%JOBS%"=="" set "JOBS=8"

call "<vs-buildtools>\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 10
set "PATH=%CUDAROOT%\bin;%PATH%"
set "CUDA_PATH=%CUDAROOT%"

"%CMAKE%" -G Ninja "-DCMAKE_MAKE_PROGRAM=%NINJA%" -S "%SRC%" -B "%BUILD%" -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF ^
  -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_TESTS=OFF ^
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON ^
  "-DCMAKE_CUDA_COMPILER=%CUDAROOT%\bin\nvcc.exe" "-DCUDAToolkit_ROOT=%CUDAROOT%" ^
  "-DCMAKE_CXX_FLAGS=/I%CUDAROOT%\include\targets\x64" ^
  "-DCMAKE_CUDA_FLAGS=-I%CUDAROOT%\include\targets\x64" || exit /b 20

"%CMAKE%" --build "%BUILD%" --target llama-server llama-cli llama-bench -j %JOBS% || exit /b 30

rem the copy probe (tools\copy_probe.cu), same nvcc
"%CUDAROOT%\bin\nvcc.exe" -O2 -arch=sm_89 -I"%CUDAROOT%\include\targets\x64" -o "%BUILD%\bin\copy_probe.exe" "%R%\tools\copy_probe.cu" || exit /b 35
"%CUDAROOT%\bin\nvcc.exe" -O2 -arch=sm_89 -I"%CUDAROOT%\include\targets\x64" -o "%BUILD%\bin\ballast.exe" "%R%\tools\ballast.cu" || exit /b 36

"%BUILD%\bin\llama-cli.exe" --version || exit /b 40
echo BUILD OK
