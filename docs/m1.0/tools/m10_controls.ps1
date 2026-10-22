# Marian M1.0 controls and floor candidates, after the Swift campaign. Each model: full 16 GB GPU, then the 8 GiB
# floor (VRAM ballast 8184 MiB on the 16,376 MiB card, 32 GB RAM real). Cache arms carry the P0 trace.
$ErrorActionPreference = 'Continue'
$R = '<research>'
$L = '<models>'
$out = "$R\m1.0\controls"; New-Item -ItemType Directory -Force $out | Out-Null
$log = "$out\controls.log"
function Arm($tag, $model, $args_, $extra) {
    "$(Get-Date -Format o) START $tag" | Out-File -Append -Encoding utf8 $log
    $p = @{ Tag = $tag; Model = $model; ServerArgs = $args_; OutDir = $out } + $extra
    & "$R\tools\run_arm.ps1" @p 2>&1 | Out-File -Append -Encoding utf8 $log
    "$(Get-Date -Format o) END $tag" | Out-File -Append -Encoding utf8 $log
}
$floor = @{ BallastMiB = 8184; DecodeN = 128 }
$ph16 = @{ Ctx = 16384; Phases = 'warm,decode,pp4k,post' }        # Alder's fields are at ctx 16K

# (1) Gemma 4 26B-A4B IQ4_XS, 30 layers, 128 experts, 8 used
$G = "$L\gemma-4-26B-A4B-it-UD-IQ4_XS.gguf"
Arm 'gemma-full-ncmoe0' $G '-ncmoe 0' $ph16
Arm 'gemma-floor-ncmoe24' $G '-ncmoe 24' ($ph16 + $floor)
Arm 'gemma-floor-c2500' $G '-ncmoe 30 --moe-cache-mib 2500' ($ph16 + $floor)

# (2) Qwen3.8-Flash-Next Coder IQ1_M (half the experts), if downloaded and verified
$C = "$L\qwen38-coder\Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf"
if ((Test-Path "$C.ok") -and (Test-Path ($C -replace '00001-of', '00002-of'))) {
    Arm 'coder-full-c9400' $C '-ncmoe 48 --moe-cache-mib 9400' @{}
    Arm 'coder-floor-c2400' $C '-ncmoe 48 --moe-cache-mib 2400' $floor
    Arm 'coder-floor-ncmoe45' $C '-ncmoe 45' $floor
}

# (3) Qwen3.6-35B-A3B UD-Q4_K_XL (geometry shared by its post-trains), 40 layers, 256 experts, 8 used
$Q = "$L\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
Arm 'q36-full-c9000' $Q '-ncmoe 40 --moe-cache-mib 9000' $ph16
Arm 'q36-floor-c2500' $Q '-ncmoe 40 --moe-cache-mib 2500' ($ph16 + $floor)
Arm 'q36-floor-ncmoe32' $Q '-ncmoe 32' ($ph16 + $floor)

# (4) Qwen3.8-27B dense UD-Q3_K_XL, the non-MoE reference: partial offload on 8 GiB
$D = "$L\Qwen3.8-27B-UD-Q3_K_XL.gguf"
Arm 'dense27-full' $D '' $ph16
Arm 'dense27-floor-ngl40' $D '-ngl 40' ($ph16 + $floor)

python -I "$R\tools\analyse_m10.py" $out "$R\m1.0\controls-summary.md" 2>&1 | Out-File -Append -Encoding utf8 $log
"$(Get-Date -Format o) CONTROLS DONE" | Out-File -Append -Encoding utf8 $log
