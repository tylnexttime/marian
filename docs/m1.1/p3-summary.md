| arm | decode tok/s: median / all tokens | post-pp16K tok/s: median / all tokens | pp4k tok/s | pp16K tok/s | hit median / p10 | MiB up/token | wait / host ms per token | VRAM peak | free RAM min | peak WS / commit GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| p3-r1-c9400 | 7.3 / 6.9 | 5.0 / 4.5 | 53.3 | 119.3 | 90.8% / 80.4% | 69.3 | 0.94 / 75.37 | 14991 | 0.23 | 22.07 / 20.14 |
| p3-r1-c9400-ring | 12.4 / 12.0 | 10.7 / 9.8 | 45.0 | 105.0 | 90.8% / 80.4% | 69.3 | 0.77 / 48.01 | 15021 | 0.14 | 21.78 / 20.26 |
| p3-r1-c9400-tier6g | 14.7 / 14.2 | 14.8 / 13.8 | 25.5 | 34.0 | 90.8% / 80.4% | 69.3 | 0.77 / 39.11 | 15021 | 0.11 | 21.48 / 26.27 |
| p3-r1-stock34 | 8.6 / 7.7 | 4.2 / 4.2 | 104.1 | 145.2 | - / - | - | - / - | 15863 | 0.16 | 21.81 / 20.84 |
| p3-r2-c9400 | 6.3 / 6.2 | 4.6 / 4.2 | 47.2 | 112.0 | 90.8% / 80.4% | 69.3 | 1.06 / 83.91 | 14990 | 0.21 | 21.57 / 20.13 |
| p3-r2-c9400-ring | 12.0 / 12.0 | 11.7 / 10.8 | 41.0 | 109.9 | 90.8% / 80.4% | 69.3 | 0.76 / 47.91 | 15020 | 0.13 | 21.75 / 20.25 |
| p3-r2-c9400-tier6g | 14.0 / 14.1 | 14.4 / 13.2 | 23.0 | 35.8 | 90.8% / 80.4% | 69.3 | 0.76 / 39.00 | 15020 | 0.10 | 20.73 / 26.27 |
| p3-r2-stock34 | 11.0 / 8.9 | 3.8 / 3.8 | 96.1 | 136.4 | - / - | - | - / - | 15728 | 0.11 | 21.77 / 20.83 |

Speed excludes requests that stopped within 8 tokens (counted here): p3-r1-c9400 decode 1, p3-r1-c9400 post 1, p3-r1-c9400-ring decode 1, p3-r1-c9400-ring post 1, p3-r1-c9400-tier6g decode 1, p3-r1-c9400-tier6g post 1, p3-r1-stock34 decode 1, p3-r1-stock34 post 1, p3-r2-c9400 decode 1, p3-r2-c9400 post 1, p3-r2-c9400-ring decode 1, p3-r2-c9400-ring post 1, p3-r2-c9400-tier6g decode 1, p3-r2-c9400-tier6g post 1, p3-r2-stock34 decode 1, p3-r2-stock34 post 1

| arm A | arm B | decode texts identical | top-3 logprobs identical | max abs dlogprob, chosen token | compared over |
|---|---|---|---|---|---|
| p3-r1-c9400 | p3-r1-c9400-ring | 10/10 | not measured | - | whole text |
| p3-r1-c9400 | p3-r1-c9400-tier6g | 10/10 | not measured | - | whole text |
| p3-r1-c9400 | p3-r1-stock34 | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| p3-r1-c9400 | p3-r2-c9400 | 10/10 | not measured | - | whole text |
| p3-r1-c9400 | p3-r2-c9400-ring | 10/10 | not measured | - | whole text |
| p3-r1-c9400 | p3-r2-c9400-tier6g | 10/10 | not measured | - | whole text |
| p3-r1-c9400 | p3-r2-stock34 | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| p3-r1-c9400-ring | p3-r1-c9400-tier6g | 10/10 | not measured | - | whole text |
| p3-r1-c9400-ring | p3-r1-stock34 | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| p3-r1-c9400-ring | p3-r2-c9400 | 10/10 | not measured | - | whole text |
| p3-r1-c9400-ring | p3-r2-c9400-ring | 10/10 | not measured | - | whole text |
| p3-r1-c9400-ring | p3-r2-c9400-tier6g | 10/10 | not measured | - | whole text |
| p3-r1-c9400-ring | p3-r2-stock34 | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| p3-r1-c9400-tier6g | p3-r1-stock34 | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| p3-r1-c9400-tier6g | p3-r2-c9400 | 10/10 | not measured | - | whole text |
| p3-r1-c9400-tier6g | p3-r2-c9400-ring | 10/10 | not measured | - | whole text |
| p3-r1-c9400-tier6g | p3-r2-c9400-tier6g | 10/10 | not measured | - | whole text |
| p3-r1-c9400-tier6g | p3-r2-stock34 | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| p3-r1-stock34 | p3-r2-c9400 | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| p3-r1-stock34 | p3-r2-c9400-ring | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| p3-r1-stock34 | p3-r2-c9400-tier6g | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| p3-r1-stock34 | p3-r2-stock34 | 10/10 | not measured | - | whole text |
| p3-r2-c9400 | p3-r2-c9400-ring | 10/10 | not measured | - | whole text |
| p3-r2-c9400 | p3-r2-c9400-tier6g | 10/10 | not measured | - | whole text |
| p3-r2-c9400 | p3-r2-stock34 | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| p3-r2-c9400-ring | p3-r2-c9400-tier6g | 10/10 | not measured | - | whole text |
| p3-r2-c9400-ring | p3-r2-stock34 | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| p3-r2-c9400-tier6g | p3-r2-stock34 | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
