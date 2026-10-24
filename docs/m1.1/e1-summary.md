| arm | decode tok/s | post-pp16K tok/s | pp4k tok/s | pp16K tok/s | hit median / p10 | MiB up/token | wait / host ms per token | VRAM peak | free RAM min | peak WS / commit GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| e1-c4000 | 7.1 | - | - | - | 74.4% / 56.9% | 178.7 | 0.87 / 81.40 | 9460 | 0.23 | 22.8 / 13.06 |
| e1-c9400 | 7.5 | - | - | - | 90.8% / 80.4% | 69.2 | 0.91 / 69.30 | 14860 | 0.25 | 22.58 / 18.34 |

| arm A | arm B | decode texts identical | top-3 probs identical |
|---|---|---|---|
| e1-c4000 | e1-c9400 | 10/10 | 10/10 |
