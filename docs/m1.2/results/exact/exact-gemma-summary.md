| arm | decode tok/s: median / all tokens | post-pp16K tok/s: median / all tokens | pp4k tok/s | pp16K tok/s | hit median / p10 | MiB up/token | wait / host ms per token | VRAM peak | free RAM min | peak WS / commit GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| g-c | 12.9 / 11.9 | - | - | - | - / - | - | - / - | 6730 | 6.92 | 13.51 / 7.21 |
| g-f1 | 83.3 / 82.8 | - | - | - | - / - | - | - / - | 14553 | 7.05 | 13.71 / 15.05 |
| g-f2 | 84.9 / 84.4 | - | - | - | - / - | - | - / - | 14752 | 6.68 | 13.71 / 15.06 |
| g-m | 34.0 / 34.5 | - | - | - | 83.3% / 66.2% | 117.4 | 0.40 / 12.13 | 6818 | 7.21 | 13.47 / 7.49 |
| g-r | 12.2 / 12.7 | - | - | - | 83.3% / 66.2% | 117.4 | 0.58 / 46.56 | 7054 | 12.87 | 7.28 / 7.55 |
| g-t | 23.5 / 24.2 | - | - | - | 83.3% / 66.2% | 117.4 | 0.45 / 17.99 | 6789 | 9.75 | 11.28 / 11.56 |

Speed excludes requests that stopped within 8 tokens (counted here): none

| arm A | arm B | decode texts identical | top-3 logprobs identical | max abs dlogprob, chosen token | compared over |
|---|---|---|---|---|---|
| g-c | g-f1 | 0/10 | 0/10 | 0.59 | whole text |
| g-c | g-f2 | 0/10 | 0/10 | 0.59 | whole text |
| g-c | g-m | 0/10 | 0/10 | 0.59 | whole text |
| g-c | g-r | 0/10 | 0/10 | 0.59 | whole text |
| g-c | g-t | 0/10 | 0/10 | 0.59 | whole text |
| g-f1 | g-f2 | 10/10 | 10/10 | 0 | whole text |
| g-f1 | g-m | 10/10 | 10/10 | 0 | whole text |
| g-f1 | g-r | 10/10 | 10/10 | 0 | whole text |
| g-f1 | g-t | 10/10 | 10/10 | 0 | whole text |
| g-f2 | g-m | 10/10 | 10/10 | 0 | whole text |
| g-f2 | g-r | 10/10 | 10/10 | 0 | whole text |
| g-f2 | g-t | 10/10 | 10/10 | 0 | whole text |
| g-m | g-r | 10/10 | 10/10 | 0 | whole text |
| g-m | g-t | 10/10 | 10/10 | 0 | whole text |
| g-r | g-t | 10/10 | 10/10 | 0 | whole text |
