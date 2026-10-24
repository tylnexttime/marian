| arm | decode tok/s: median / all tokens | post-pp16K tok/s: median / all tokens | pp4k tok/s | pp16K tok/s | hit median / p10 | MiB up/token | wait / host ms per token | VRAM peak | free RAM min | peak WS / commit GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| e1-c4000 | 7.2 / 7.0 | - | - | - | 74.4% / 56.9% | 178.7 | 0.87 / 81.40 | 9460 | 0.23 | 22.8 / 13.06 |
| e1-c9400 | 7.8 / 7.3 | - | - | - | 90.8% / 80.4% | 69.2 | 0.91 / 69.30 | 14860 | 0.25 | 22.58 / 18.34 |

Speed excludes requests that stopped within 8 tokens (counted here): e1-c4000 decode 1, e1-c9400 decode 1

| arm A | arm B | decode texts identical | top-3 logprobs identical | max abs dlogprob, chosen token | compared over |
|---|---|---|---|---|---|
| e1-c4000 | e1-c9400 | 10/10 | not measured | - | whole text |
