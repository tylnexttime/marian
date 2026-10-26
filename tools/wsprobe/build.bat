@echo off
rem Copyright 2026 tylnexttime (Chris Tyl and Claude Tyl) and contributors
rem SPDX-License-Identifier: Apache-2.0
rem Build wsprobe.exe and wsprobe_target.exe with MSVC. Run from a Developer prompt (cl on PATH), or set VCVARS64 to
rem the full path of vcvars64.bat and this script calls it first. Outputs go next to this file.
setlocal
if defined VCVARS64 call "%VCVARS64%" >nul || exit /b 10
where cl >nul 2>nul || (echo cl.exe not found: run from a Developer prompt or set VCVARS64 & exit /b 11)
cd /d "%~dp0"
cl /nologo /O2 /W4 /wd4996 wsprobe.c /Fe:wsprobe.exe /link psapi.lib kernel32.lib || exit /b 20
cl /nologo /O2 /W4 wsprobe_target.c /Fe:wsprobe_target.exe || exit /b 21
del /q *.obj 2>nul
echo built
