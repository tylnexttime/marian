@echo off
rem Marian M1.1: llama.cpp branch marian/m1.1 (63b4b303 + P0 trace + M1.1 patches; the M1.1 runs used a37cc461)
rem into build-m1.1, so the measured M1.0 binaries in <research>\build stay untouched. Same toolchain and network-off
rem switches as build.bat.
rem Usage: build_m11.bat [jobs=8] [probe|selftest]
rem   probe    builds only tools\replay_probe.cu (nvcc), not llama.cpp
rem   selftest builds only tools\p3_selftest.cpp (cl, CPU only; needs a finished llama.cpp build for ggml-base/ggml-cpu)
rem Never run while <research>\m1.0\GPU-BUSY exists (a build uses every core and skews CPU-timed runs).
setlocal
set "R=<research>"
set "SRC=%R%\llama.cpp"
set "BUILD=%R%\build-m1.1"
set "CUDAROOT=<maya-build>\env\Library"
set "CMAKE=<vs-buildtools>\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA=<vs-buildtools>\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
set "JOBS=%~1"
if "%JOBS%"=="" set "JOBS=8"

if exist "%R%\m1.0\GPU-BUSY" (
  echo GPU-BUSY exists: not building
  exit /b 5
)

call "<vs-buildtools>\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 10
set "PATH=%CUDAROOT%\bin;%PATH%"
set "CUDA_PATH=%CUDAROOT%"
if not exist "%BUILD%\bin" mkdir "%BUILD%\bin"

if /i "%~2"=="probe" goto probe
if /i "%~2"=="selftest" goto selftest

"%CMAKE%" -G Ninja "-DCMAKE_MAKE_PROGRAM=%NINJA%" -S "%SRC%" -B "%BUILD%" -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF ^
  -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_TESTS=OFF ^
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON ^
  "-DCMAKE_CUDA_COMPILER=%CUDAROOT%\bin\nvcc.exe" "-DCUDAToolkit_ROOT=%CUDAROOT%" ^
  "-DCMAKE_CXX_FLAGS=/I%CUDAROOT%\include\targets\x64" ^
  "-DCMAKE_CUDA_FLAGS=-I%CUDAROOT%\include\targets\x64" || exit /b 20

"%CMAKE%" --build "%BUILD%" --target llama-server llama-cli llama-bench -j %JOBS% || exit /b 30

rem no "--version" run here: it loads the CUDA backend, and the GPU may belong to a measurement
if not exist "%BUILD%\bin\llama-server.exe" exit /b 40

:probe
rem the trace-replay probe (tools\replay_probe.cu), same nvcc; standalone, links only cudart and kernel32
if exist "%R%\tools\replay_probe.cu" (
  "%CUDAROOT%\bin\nvcc.exe" -O2 -std=c++17 -arch=sm_89 -I"%CUDAROOT%\include\targets\x64" -o "%BUILD%\bin\replay_probe.exe" "%R%\tools\replay_probe.cu" || exit /b 35
)
if /i "%~2"=="probe" goto done

:selftest
rem CPU-only self-test of the P3 miss source (src\llama-moe-source.cpp): no CUDA, the ggml CPU backend plays the device
if not exist "%BUILD%\obj-selftest" mkdir "%BUILD%\obj-selftest"
cl /nologo /O2 /EHsc /std:c++17 /I"%SRC%\include" /I"%SRC%\src" /I"%SRC%\ggml\include" "%R%\tools\p3_selftest.cpp" "%SRC%\src\llama-moe-source.cpp" ^
  /Fo"%BUILD%\obj-selftest\\" /Fe"%BUILD%\bin\p3_selftest.exe" /link "%BUILD%\ggml\src\ggml-base.lib" "%BUILD%\ggml\src\ggml-cpu.lib" || exit /b 36

:done
echo BUILD OK
