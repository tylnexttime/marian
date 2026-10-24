# Marian, Saturday 24/10/2026: the design-9.3 confirmation of the M1 decode exit for the P3 tier.
# E40 (two rounds) plus the corrected analysis (requests that stop within 8 tokens excluded) put the 6 GiB tier at 13-15
# tok/s warm AND post-pp16K, above the M1 exit's 12. Two rounds are indicative only; 9.3 asks for >= 5 interleaved
# rounds, counterbalanced, fresh process each, page-cache state logged. Arms (same as E40):
#   T = M1.1 build, -ncmoe 48 --moe-cache-mib 9400 --moe-miss-source tier --moe-tier-mib 6144
#   S = M1.0 build, -ncmoe 34 (stock reference)
# Order: T S | S T | T S | S T | T S  (each arm 5 times). Outputs m1.2\confirm. Holds GPU-BUSY.
$ErrorActionPreference = 'Continue'
$R = '<research>'
$busy = "$R\m1.0\GPU-BUSY"
$log = "$R\m1.0\logs\confirm.log"
$out = "$R\m1.2\confirm"
function L($s) { "$(Get-Date -Format o) $s" | Out-File -Append -Encoding utf8 $log }
while (Test-Path $busy) { Start-Sleep 30 }
for ($i = 0; $i -lt 360; $i++) {
    if (-not (Get-Process -Name cl, nvcc, ninja, cmake, link, cicc, ptxas -ErrorAction SilentlyContinue)) { break }
    Start-Sleep 10
}
"confirm $(Get-Date -Format o)" | Out-File -Encoding utf8 $busy
New-Item -ItemType Directory -Force $out | Out-Null
L 'CONFIRM START'
$T = @{ s = 'run_arm_m11.ps1'; t = 'tier6g'; a = '-ncmoe 48 --moe-cache-mib 9400 --moe-miss-source tier --moe-tier-mib 6144' }
$S = @{ s = 'run_arm.ps1'; t = 'stock34'; a = '-ncmoe 34' }
$rounds = @(@($T, $S), @($S, $T), @($T, $S), @($S, $T), @($T, $S))
for ($k = 0; $k -lt $rounds.Count; $k++) {   # not $r: PowerShell names are case-insensitive, $r IS $R
    foreach ($x in $rounds[$k]) {
        $tag = "c$($k + 1)-$($x.t)"
        L "START $tag"
        & "$R\tools\$($x.s)" -Tag $tag -ServerArgs $x.a -OutDir $out 2>&1 | ForEach-Object { L "arm: $_" }
        L "END $tag"
    }
}
Remove-Item -Force $busy -ErrorAction SilentlyContinue
python -I "$R\tools\analyse_m10.py" $out "$R\m1.2\confirm-summary.md" 2>&1 | Out-Null
L 'CONFIRM DONE'
Get-Content "$R\m1.2\confirm-summary.md" -TotalCount 14
