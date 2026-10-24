# Marian, Saturday 24/10/2026: E45, logprob exactness with a client that finally records probabilities.
# Until today run_arm*.ps1 read top_probs/prob, which this server does not send (it sends top_logprobs/logprob), so every
# top-3 comparison since 22/10 compared nulls. Fixed; this chain re-measures exactness on build-m1.1 (the version we
# publish). No ballast: exactness, not speed. Phases warm + decode, 10 prompts x 128 tokens, n_probs 3.
#   Gemma 26B: F1 all-VRAM | M cache mmap | T cache + tier | R cache + ring | C stock -ncmoe 24 (CPU experts) | F2 = F1
#   Swift 1.5: SM cache mmap | ST cache + tier | SS stock -ncmoe 34
# A smoke arm first: if it did not record numeric logprobs the chain stops before spending the GPU. Holds GPU-BUSY.
$ErrorActionPreference = 'Continue'
$R = '<research>'
$busy = "$R\m1.0\GPU-BUSY"
$log = "$R\m1.0\logs\exact.log"
$out = "$R\m1.2\exact"
$smoke = "$R\m1.2\exact-smoke"
$gemma = '<models>\gemma-4-26B-A4B-it-UD-IQ4_XS.gguf'
function L($s) { "$(Get-Date -Format o) $s" | Out-File -Append -Encoding utf8 $log }
while (Test-Path $busy) { Start-Sleep 30 }
"exact $(Get-Date -Format o)" | Out-File -Encoding utf8 $busy
New-Item -ItemType Directory -Force $out, $smoke | Out-Null
L 'EXACT START'

& "$R\tools\run_arm_m11.ps1" -Tag 'smoke-f1' -ServerArgs '-ncmoe 0' -Model $gemma -Ctx 16384 -Phases 'decode' -DecodeN 8 -OutDir $smoke 2>&1 | ForEach-Object { L "arm: $_" }
$check = python -I -c @"
import json, sys
rs = [json.loads(l) for l in open(r'$smoke\smoke-f1.requests.jsonl', encoding='utf-8-sig')]
bad = [r['idx'] for r in rs if not r.get('ok') or not r.get('probs') or not all(isinstance(x.get('lp'), (int, float)) and x.get('top') and all(isinstance(e[1], (int, float)) for e in x['top']) for x in r['probs'])]
print('SMOKE OK' if rs and not bad else f'SMOKE BAD {bad} ' + str(rs[0])[:300])
"@
L "smoke: $check"
if ($check -notlike 'SMOKE OK*') { L 'EXACT ABORT: the client still records no logprobs'; Remove-Item -Force $busy; exit 1 }

$arms = @(
    @{ t = 'g-f1'; a = '-ncmoe 0'; m = $gemma; c = 16384 },
    @{ t = 'g-m'; a = '-ncmoe 30 --moe-cache-mib 2500'; m = $gemma; c = 16384 },
    @{ t = 'g-t'; a = '-ncmoe 30 --moe-cache-mib 2500 --moe-miss-source tier --moe-tier-mib 4096'; m = $gemma; c = 16384 },
    @{ t = 'g-r'; a = '-ncmoe 30 --moe-cache-mib 2500 --moe-miss-source ring'; m = $gemma; c = 16384 },
    @{ t = 'g-c'; a = '-ncmoe 24'; m = $gemma; c = 16384 },
    @{ t = 'g-f2'; a = '-ncmoe 0'; m = $gemma; c = 16384 }
)
foreach ($x in $arms) {
    L "START $($x.t)"
    & "$R\tools\run_arm_m11.ps1" -Tag $x.t -ServerArgs $x.a -Model $x.m -Ctx $x.c -Phases 'warm,decode' -DecodeN 128 -OutDir "$out\gemma" 2>&1 | ForEach-Object { L "arm: $_" }
    L "END $($x.t)"
}
$swift = @(
    @{ t = 's-m'; a = '-ncmoe 48 --moe-cache-mib 9400' },
    @{ t = 's-t'; a = '-ncmoe 48 --moe-cache-mib 9400 --moe-miss-source tier --moe-tier-mib 6144' },
    @{ t = 's-s'; a = '-ncmoe 34' }
)
foreach ($x in $swift) {
    L "START $($x.t)"
    & "$R\tools\run_arm_m11.ps1" -Tag $x.t -ServerArgs $x.a -Phases 'warm,decode' -DecodeN 128 -OutDir "$out\swift" 2>&1 | ForEach-Object { L "arm: $_" }
    L "END $($x.t)"
}
Remove-Item -Force $busy -ErrorAction SilentlyContinue
python -I "$R\tools\analyse_m10.py" "$out\gemma" "$R\m1.2\exact-gemma-summary.md" 2>&1 | Out-Null
python -I "$R\tools\analyse_m10.py" "$out\swift" "$R\m1.2\exact-swift-summary.md" 2>&1 | Out-Null
L 'EXACT DONE'
