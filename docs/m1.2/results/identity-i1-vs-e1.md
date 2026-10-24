| arm | decode tok/s: median / all tokens | post-pp16K tok/s: median / all tokens | pp4k tok/s | pp16K tok/s | hit median / p10 | MiB up/token | wait / host ms per token | VRAM peak | free RAM min | peak WS / commit GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| e1-c9400 | 7.8 / 7.3 | - | - | - | - / - | - | - / - | - | - | - / - |
| i1-c9400 | 6.6 / 6.3 | - | - | - | - / - | - | - / - | - | - | - / - |

Speed excludes requests that stopped within 8 tokens (counted here): e1-c9400 decode 1, i1-c9400 decode 1

| arm A | arm B | decode texts identical | top-3 logprobs identical | max abs dlogprob, chosen token | compared over |
|---|---|---|---|---|---|
| e1-c9400 | i1-c9400 | 10/10 | not measured | - | whole text |
