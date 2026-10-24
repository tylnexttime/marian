| arm | decode tok/s: median / all tokens | post-pp16K tok/s: median / all tokens | pp4k tok/s | pp16K tok/s | hit median / p10 | MiB up/token | wait / host ms per token | VRAM peak | free RAM min | peak WS / commit GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| i1-c9400 | 6.6 / 6.3 | - | - | - | 90.8% / 80.4% | 69.2 | 1.02 / 84.19 | 14892 | 0.21 | 21.37 / 18.34 |
| i2-c9400 | 7.0 / 6.8 | 4.6 / 4.2 | 51.5 | 101.4 | 90.8% / 80.4% | 69.3 | 0.94 / 79.41 | 14987 | 0.23 | 21.33 / 20.14 |

Speed excludes requests that stopped within 8 tokens (counted here): i1-c9400 decode 1, i2-c9400 decode 1, i2-c9400 post 1

| arm A | arm B | decode texts identical | top-3 probs identical | compared over |
|---|---|---|---|---|
| i1-c9400 | i2-c9400 | 10/10 | 10/10 | whole text |
