# Requested check (b): Maya-S24 on C: (SK hynix PC801) vs E: (PNY CS2241). Copy with timestamps (Maya's pack checks
# mtime_ns), run Maya's standard --bench in the order E, C, C, E, each with the resource sampler, then delete the C: copy.
$ErrorActionPreference = 'Continue'
$M = '<maya-build>\src\project-maya-1.1.19'
$out = '<research>\m1.0\maya-cve'; New-Item -ItemType Directory -Force $out | Out-Null
$log = "$out\maya-cve.log"
$env:PATH = '<maya-build>\env\Library\bin;' + $env:PATH
$env:MAYA_UPDATE_CHECK = '0'

$cfgE = Get-Content -Raw "$M\maya-maya-s24.json" | ConvertFrom-Json
$cfgC = Get-Content -Raw "$M\maya-maya-s24.json" | ConvertFrom-Json
$i = [array]::IndexOf($cfgC.args, '--glm-pack'); $cfgC.args[$i + 1] = '<system-ssd>\maya-models\Maya-S24\pack'
$cfgC.tokenizer = '<system-ssd>\maya-models\Maya-S24\pack\tokenizer'
$cfgC.log = "$M\maya-maya-s24-C.log"
($cfgC | ConvertTo-Json -Depth 6) | Out-File -Encoding utf8 "$M\maya-maya-s24-C.json"

$t0 = Get-Date
robocopy '<data-ssd>\maya-models\Maya-S24' '<system-ssd>\maya-models\Maya-S24' /E /COPY:DAT /DCOPY:T /NP /NFL /NDL /R:2 /W:5 | Out-File -Encoding utf8 "$out\robocopy.txt"
"copy E: to C: took {0:F0} s, robocopy rc {1}" -f ((Get-Date) - $t0).TotalSeconds, $LASTEXITCODE | Out-File -Append -Encoding utf8 $log

function Bench($tag, $cfg) {
    $stop = "$out\$tag.stop"; Remove-Item -Force $stop -ErrorAction SilentlyContinue
    $res = Start-Process -PassThru -WindowStyle Hidden pwsh -ArgumentList '-NoProfile', '-File', '<baseline>\tools\reslog.ps1', '-Out', "$out\$tag.res.csv", '-StopFile', $stop
    $a = Get-Date
    "$(Get-Date -Format o) START $tag" | Out-File -Append -Encoding utf8 $log
    cmd /c "`"$M\START-MAYA.bat`" --bench --plain --config `"$cfg`" < NUL" *> "$out\$tag.console.txt"
    "$(Get-Date -Format o) END $tag rc $LASTEXITCODE wall {0:F0} s" -f ((Get-Date) - $a).TotalSeconds | Out-File -Append -Encoding utf8 $log
    Copy-Item -Force "$M\maya-bench.txt" "$out\$tag.maya-bench.txt" -ErrorAction SilentlyContinue
    New-Item -ItemType File $stop | Out-Null
    if (-not $res.WaitForExit(20000)) { Stop-Process -Id $res.Id -Force }
}
Bench 'E1' "$M\maya-maya-s24.json"
Bench 'C1' "$M\maya-maya-s24-C.json"
Bench 'C2' "$M\maya-maya-s24-C.json"
Bench 'E2' "$M\maya-maya-s24.json"

Remove-Item -Recurse -Force '<system-ssd>\maya-models\Maya-S24'
if (-not (Get-ChildItem '<system-ssd>\maya-models' -ErrorAction SilentlyContinue)) { Remove-Item -Force '<system-ssd>\maya-models' -ErrorAction SilentlyContinue }
Remove-Item -Force "$M\maya-maya-s24-C.json"
"$(Get-Date -Format o) C: copy deleted; free C: {0:N0} GB" -f ((Get-PSDrive C).Free / 1GB) | Out-File -Append -Encoding utf8 $log
"MAYA CVE DONE" | Out-File -Append -Encoding utf8 $log
