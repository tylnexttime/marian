param(
    [Parameter(Mandatory)] [string]$Tag,          # output name, e.g. r1-cache8000
    [Parameter(Mandatory)] [string]$ServerArgs,   # placement flags, e.g. "-ncmoe 48 --moe-cache-mib 8000"
    [string]$Model = '<models>\Swift-1.5-IQ2_XS\Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf',
    [int]$Ctx = 20480,
    [string]$Phases = 'warm,decode,pp4k,pp16k,post',
    [int]$DecodeN = 256,
    [int]$Port = 8090,
    [int]$BallastMiB = 0,                          # >0: hold this much VRAM first, to simulate a smaller GPU
    [string]$OutDir = '<research>\m1.0\runs'   # where the arm's files go
)
# Marian M1.0 arm: one llama-server (127.0.0.1:$Port) with the P0 trace on, a fixed request sequence, resources sampled.
# Outputs in <research>\m1.0\runs\: <Tag>.requests.jsonl, .trace.csv, .server.log, .res.csv, .proc.json
$ErrorActionPreference = 'Stop'
$R = '<research>'
$out = $OutDir; New-Item -ItemType Directory -Force $out | Out-Null
$env:PATH = '<maya-build>\env\Library\bin;' + $env:PATH
$env:MARIAN_MOE_TRACE = "$out\$Tag.trace.csv"
$prompts = Get-Content -Raw -Encoding utf8 "$R\m1.0\prompts.json" | ConvertFrom-Json
$freq = [System.Diagnostics.Stopwatch]::Frequency
function Now-Us { [long]([System.Diagnostics.Stopwatch]::GetTimestamp() * 1000000.0 / $freq) }   # same clock as steady_clock (QPC)

if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { throw "port $Port is busy" }
$stop = "$out\$Tag.stop"; Remove-Item -Force $stop -ErrorAction SilentlyContinue
$res = Start-Process -PassThru -WindowStyle Hidden pwsh -ArgumentList '-NoProfile', '-File', '<baseline>\tools\reslog.ps1', '-Out', "$out\$Tag.res.csv", '-StopFile', $stop
$bal = $null
if ($BallastMiB -gt 0) {
    $bal = Start-Process -PassThru -WindowStyle Hidden "$R\build\bin\ballast.exe" -ArgumentList "$BallastMiB", $stop -RedirectStandardOutput "$out\$Tag.ballast.txt"
    for ($i = 0; $i -lt 60 -and -not (Select-String -Quiet -Path "$out\$Tag.ballast.txt" -Pattern 'held' -ErrorAction SilentlyContinue); $i++) { Start-Sleep 1 }
    if ($bal.HasExited) { throw "ballast failed: $(Get-Content "$out\$Tag.ballast.txt")" }
}

$sargs = @('-m', $Model, '--host', '127.0.0.1', '--port', "$Port", '-c', "$Ctx", '-ngl', '99', '-fa', 'on', '-t', '16',
          '--parallel', '1', '--no-webui') + ($ServerArgs -split ' ' | Where-Object { $_ })
"CMD: llama-server $($sargs -join ' ')" | Out-File -Encoding utf8 "$out\$Tag.cmd.txt"
$t_load0 = Get-Date
$srv = Start-Process -PassThru -WindowStyle Hidden "$R\build\bin\llama-server.exe" -ArgumentList $sargs `
        -RedirectStandardError "$out\$Tag.server.log" -RedirectStandardOutput "$out\$Tag.server.out"
$base = "http://127.0.0.1:$Port"
$ready = $false
for ($i = 0; $i -lt 900 -and -not $srv.HasExited; $i++) {
    try { if ((Invoke-WebRequest -UseBasicParsing -TimeoutSec 5 "$base/health").StatusCode -eq 200) { $ready = $true; break } } catch {}
    Start-Sleep -Seconds 1
}
$load_s = ((Get-Date) - $t_load0).TotalSeconds
$log = "$out\$Tag.requests.jsonl"; Remove-Item -Force $log -ErrorAction SilentlyContinue

function Ask([string]$phase, [int]$idx, [string]$prompt, [int]$n, [int]$nprobs) {
    $body = @{ prompt = $prompt; n_predict = $n; temperature = 0; top_k = 1; seed = 42; cache_prompt = $false; n_probs = $nprobs } | ConvertTo-Json -Compress
    $a = Now-Us
    try {
        $r = Invoke-RestMethod -Method Post -TimeoutSec 3600 -ContentType 'application/json; charset=utf-8' "$base/completion" -Body ([Text.Encoding]::UTF8.GetBytes($body))
        $b = Now-Us
        $probs = $null
        if ($nprobs -gt 0 -and $r.completion_probabilities) {
            # This server sends top_logprobs/logprob (pre-sampling) unless post_sampling_probs is set; until 24/10/2026 this
            # read top_probs/prob, got nothing, and `$null | ForEach-Object` stored [[null], null] for every token, so
            # every top-3 comparison compared nulls. Read both shapes, and refuse a response that carries no top-k at all.
            $pv = { param($o) if ($null -ne $o.logprob) { [double]$o.logprob } elseif ($null -ne $o.prob) { [double]$o.prob } else { $null } }
            $probs = @($r.completion_probabilities | ForEach-Object {
                $tl = if ($null -ne $_.top_logprobs) { $_.top_logprobs } else { $_.top_probs }
                @{ id = $_.id; lp = (& $pv $_); top = @($tl | Where-Object { $null -ne $_ } | ForEach-Object { , @($_.id, (& $pv $_)) }) } })
            if ($probs.Count -eq 0 -or @($probs | Where-Object { $_.top.Count -eq 0 -or $null -eq $_.lp }).Count) {
                throw "n_probs=$nprobs but the response carried no usable top-k: the probability parser reads the wrong fields" }
        }
        $rec = [ordered]@{ tag = $Tag; phase = $phase; idx = $idx; t0_us = $a; t1_us = $b; ok = $true;
            prompt_n = $r.timings.prompt_n; prompt_tps = $r.timings.prompt_per_second; gen_n = $r.timings.predicted_n;
            gen_tps = $r.timings.predicted_per_second; content = $r.content; probs = $probs }
    } catch {
        $rec = [ordered]@{ tag = $Tag; phase = $phase; idx = $idx; t0_us = $a; t1_us = (Now-Us); ok = $false; error = "$_" }
    }
    ($rec | ConvertTo-Json -Compress -Depth 8) | Out-File -Append -Encoding utf8 $log
}

if ($ready) {
    foreach ($ph in ($Phases -split ',')) {
        switch ($ph) {
            'warm'   { Ask 'warm' 0 $prompts.decode[0] 64 0 }
            'decode' { for ($k = 0; $k -lt $prompts.decode.Count; $k++) { Ask 'decode' $k $prompts.decode[$k] $DecodeN 3 } }
            'pp4k'   { Ask 'pp4k' 0 $prompts.pp4k 32 0 }
            'pp16k'  { Ask 'pp16k' 0 $prompts.pp16k 32 0 }
            'post'   { for ($k = 0; $k -lt 3; $k++) { Ask 'post' $k $prompts.decode[$k] 128 3 } }
        }
    }
}
$proc = [ordered]@{ tag = $Tag; ready = $ready; load_s = [math]::Round($load_s, 1); exited_early = $srv.HasExited }
if (-not $srv.HasExited) {
    $srv.Refresh()
    $proc.peak_working_set_gib = [math]::Round($srv.PeakWorkingSet64 / 1GB, 2)
    $proc.peak_private_commit_gib = [math]::Round($srv.PeakPagedMemorySize64 / 1GB, 2)
    $proc.working_set_gib = [math]::Round($srv.WorkingSet64 / 1GB, 2)
    $proc.private_commit_gib = [math]::Round($srv.PrivateMemorySize64 / 1GB, 2)
    Stop-Process -Id $srv.Id -Force
}
($proc | ConvertTo-Json) | Out-File -Encoding utf8 "$out\$Tag.proc.json"
New-Item -ItemType File $stop | Out-Null
if (-not $res.WaitForExit(20000)) { Stop-Process -Id $res.Id -Force }
if ($bal -and -not $bal.WaitForExit(10000)) { Stop-Process -Id $bal.Id -Force }
"ARM DONE $Tag ready=$ready"
