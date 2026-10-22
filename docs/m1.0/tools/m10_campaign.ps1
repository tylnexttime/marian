# Marian M1.0 Swift campaign: counterbalanced arms (A-B-C-D, D-C-B-A), then the 8 GB floor arms with a VRAM ballast.
# Each arm is a fresh llama-server; warm state = the fixed 64-token warm-up at the start of each arm.
$ErrorActionPreference = 'Continue'
$R = '<research>'
$log = "$R\m1.0\runs\campaign.log"
function Arm($tag, $args_, $extra) {
    "$(Get-Date -Format o) START $tag" | Out-File -Append -Encoding utf8 $log
    $p = @{ Tag = $tag; ServerArgs = $args_ } + $extra
    & "$R\tools\run_arm.ps1" @p 2>&1 | Out-File -Append -Encoding utf8 $log
    "$(Get-Date -Format o) END $tag" | Out-File -Append -Encoding utf8 $log
}
$full = @(
    @('c9400', '-ncmoe 48 --moe-cache-mib 9400'),
    @('stock34', '-ncmoe 34'),
    @('c6500', '-ncmoe 48 --moe-cache-mib 6500'),
    @('c4000', '-ncmoe 48 --moe-cache-mib 4000')
)
foreach ($a in $full) { Arm "r1-$($a[0])" $a[1] @{} }
[array]::Reverse($full)
foreach ($a in $full) { Arm "r2-$($a[0])" $a[1] @{} }

# floor: 32 GB RAM (real) + an 8 GiB GPU (ballast 8184 MiB on the 16,376 MiB card)
$floor = @{ BallastMiB = 8184; DecodeN = 128 }
Arm 'f1-c2400' '-ncmoe 48 --moe-cache-mib 2400' $floor
Arm 'f1-stock45' '-ncmoe 45' $floor
Arm 'f2-stock45' '-ncmoe 45' $floor
Arm 'f2-c2400' '-ncmoe 48 --moe-cache-mib 2400' $floor

python -I "$R\tools\analyse_m10.py" "$R\m1.0\runs" "$R\m1.0\swift-summary.md" 2>&1 | Out-File -Append -Encoding utf8 $log
"$(Get-Date -Format o) CAMPAIGN DONE" | Out-File -Append -Encoding utf8 $log
