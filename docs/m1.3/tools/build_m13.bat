@echo off
rem Marian M1.3: llama.cpp at upstream 338fce651 (63b4b303 until 28/10/2026) + the Marian patches (docs\m1.0, m1.1, m1.2 and m1.3\patches, through
rem 0009) into build-m13. The first Marian build script WITH C++ exception unwinding (/EHsc): every earlier build
rem passed CMAKE_CXX_FLAGS for the CUDA include path alone, which drops MSVC's default /EHsc. Register E57 (27/10/2026)
rem built the same sources both ways and found every answer identical (tokens, log-probabilities and top-3, max
rem difference 0, on every exactness arm), so /EHsc is the default from M1.3 on.
rem Usage: build_m13.bat [jobs=8]
rem Never run while <research>\m1.0\GPU-BUSY or <research>\GPU-IN-USE.txt exists (a build uses every core and skews
rem timed runs).
setlocal
set "R=<research>"
set "SRC=%R%\llama.cpp-m13"
set "BUILD=%R%\build-m13"
set "CUDAROOT=<maya-build>\env\Library"
set "CMAKE=<vs-buildtools>\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA=<vs-buildtools>\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
set "JOBS=%~1"
if "%JOBS%"=="" set "JOBS=8"

if exist "%R%\m1.0\GPU-BUSY" (
  echo GPU-BUSY exists: not building
  exit /b 5
)
if exist "%R%\GPU-IN-USE.txt" (
  echo GPU-IN-USE.txt exists: not building
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
  -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF -DGGML_OPENMP_FETCH=OFF -DLLAMA_BUILD_TESTS=ON ^
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON ^
  "-DCMAKE_CUDA_COMPILER=%CUDAROOT%\bin\nvcc.exe" "-DCUDAToolkit_ROOT=%CUDAROOT%" ^
  "-DCMAKE_CXX_FLAGS=/EHsc /I%CUDAROOT%\include\targets\x64" ^
  "-DCMAKE_CUDA_FLAGS=-I%CUDAROOT%\include\targets\x64" || exit /b 20

"%CMAKE%" --build "%BUILD%" --target llama-server llama-cli llama-bench -j %JOBS% || exit /b 30
echo SERVER TARGETS OK

rem the tests: ctest needs every test executable built
"%CMAKE%" --build "%BUILD%" -j %JOBS% || exit /b 31

rem no "--version" run here: it loads the CUDA backend, and the GPU may belong to a measurement
if not exist "%BUILD%\bin\llama-server.exe" exit /b 40
if not exist "%BUILD%\bin\test-moe-trace-ids.exe" exit /b 41
echo BUILD OK
