| arm | decode tok/s: median / all tokens | post-pp16K tok/s: median / all tokens | pp4k tok/s | pp16K tok/s | hit median / p10 | MiB up/token | wait / host ms per token | VRAM peak | free RAM min | peak WS / commit GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| s-m | 5.8 / 5.7 | - | - | - | 90.4% / 80.0% | 71.9 | 1.01 / 96.80 | 15106 | 0.26 | 21.67 / 18.31 |
| s-s | 7.7 / 6.3 | - | - | - | - / - | - | - / - | 15643 | 0.26 | 21.77 / 19.01 |
| s-t | 14.6 / 14.8 | - | - | - | 90.4% / 80.0% | 71.9 | 0.67 / 38.25 | 15077 | 3.89 | 17.64 / 24.46 |

Speed excludes requests that stopped within 8 tokens (counted here): s-m decode 1, s-s decode 1, s-t decode 1

| arm A | arm B | decode texts identical | top-3 logprobs identical | max abs dlogprob, chosen token | compared over |
|---|---|---|---|---|---|
| s-m | s-s | 4/10 | 0/10 | 0.337 | whole text |
| s-m | s-t | 10/10 | 10/10 | 0 | whole text |
| s-s | s-t | 4/10 | 0/10 | 0.337 | whole text |
