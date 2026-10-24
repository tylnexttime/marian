@echo off
rem Marian M1.2: llama.cpp branch marian/m1.2 (marian/m1.1 + patch 0004, the routing-id trace) into build-m1.2, from
rem its own worktree llama.cpp-m1.2, so the M1.0 binaries (build) and the M1.1 binaries (build-m1.1, being timed in the
rem P3 arms) and their source tree stay untouched. Same toolchain and switches as build_m11.bat (no /EHsc, as there).
rem Usage: build_m12.bat [jobs=8]
rem Never run while <research>\m1.0\GPU-BUSY exists (a build uses every core and skews CPU-timed runs).
setlocal
set "R=<research>"
set "SRC=%R%\llama.cpp-m1.2"
set "BUILD=%R%\build-m1.2"
set "CUDAROOT=<maya-build>\env\Library"
set "CMAKE=<vs-buildtools>\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA=<vs-buildtools>\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
set "JOBS=%~1"
if "%JOBS%"=="" set "JOBS=8"

if exist "%R%\m1.0\GPU-BUSY" (
  echo GPU-BUSY exists: not building
  exit /b 5
)
if not exist "%SRC%\src\llama-moe-trace-ids.cpp" (
  echo %SRC% has no patch 0004: see docs\m1.2\NOTES.md section 5
  exit /b 6
)

call "<vs-buildtools>\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 10
set "PATH=%CUDAROOT%\bin;%PATH%"
set "CUDA_PATH=%CUDAROOT%"
if not exist "%BUILD%\bin" mkdir "%BUILD%\bin"

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
echo BUILD OK
