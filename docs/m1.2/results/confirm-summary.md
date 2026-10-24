| arm | decode tok/s: median / all tokens | post-pp16K tok/s: median / all tokens | pp4k tok/s | pp16K tok/s | hit median / p10 | MiB up/token | wait / host ms per token | VRAM peak | free RAM min | peak WS / commit GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| c1-stock34 | 11.3 / 10.8 | 4.0 / 3.9 | 91.7 | 144.4 | - / - | - | - / - | 15750 | 0.18 | 21.72 / 20.84 |
| c1-tier6g | 14.5 / 14.5 | 14.4 / 13.4 | 18.9 | 29.7 | 90.8% / 80.4% | 69.3 | 0.72 / 37.99 | 15022 | 0.10 | 20.4 / 26.27 |
| c2-stock34 | 11.1 / 9.6 | 4.2 / 4.2 | 101.0 | 148.6 | - / - | - | - / - | 15711 | 0.22 | 22.03 / 20.83 |
| c2-tier6g | 14.9 / 14.4 | 14.7 / 13.6 | 25.8 | 34.4 | 90.8% / 80.4% | 69.3 | 0.73 / 38.30 | 15003 | 0.05 | 21.34 / 26.27 |
| c3-stock34 | 8.9 / 8.2 | 3.9 / 3.9 | 102.9 | 155.0 | - / - | - | - / - | 15724 | 0.16 | 21.88 / 20.84 |
| c3-tier6g | 14.8 / 14.6 | 14.4 / 13.4 | 25.4 | 31.2 | 90.8% / 80.4% | 69.3 | 0.70 / 37.97 | 15010 | 0.09 | 21.22 / 26.27 |
| c4-stock34 | 10.0 / 9.5 | 3.9 / 3.9 | 95.1 | 141.8 | - / - | - | - / - | 15718 | 0.21 | 21.38 / 20.83 |
| c4-tier6g | 15.1 / 14.5 | 14.7 / 13.7 | 25.3 | 34.4 | 90.8% / 80.4% | 69.3 | 0.72 / 38.24 | 15010 | 0.10 | 21.66 / 26.27 |
| c5-stock34 | 8.6 / 8.2 | 8.7 / 8.5 | 103.2 | 154.8 | - / - | - | - / - | 15719 | 0.23 | 22.08 / 20.84 |
| c5-tier6g | 14.7 / 14.6 | 14.5 / 13.5 | 22.1 | 30.3 | 90.8% / 80.4% | 69.3 | 0.71 / 38.04 | 15085 | 0.12 | 20.86 / 26.27 |

Speed excludes requests that stopped within 8 tokens (counted here): c1-stock34 decode 1, c1-stock34 post 1, c1-tier6g decode 1, c1-tier6g post 1, c2-stock34 decode 1, c2-stock34 post 1, c2-tier6g decode 1, c2-tier6g post 1, c3-stock34 decode 1, c3-stock34 post 1, c3-tier6g decode 1, c3-tier6g post 1, c4-stock34 decode 1, c4-stock34 post 1, c4-tier6g decode 1, c4-tier6g post 1, c5-stock34 decode 1, c5-stock34 post 1, c5-tier6g decode 1, c5-tier6g post 1

| arm A | arm B | decode texts identical | top-3 logprobs identical | max abs dlogprob, chosen token | compared over |
|---|---|---|---|---|---|
| c1-stock34 | c1-tier6g | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| c1-stock34 | c2-stock34 | 10/10 | not measured | - | whole text |
| c1-stock34 | c2-tier6g | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| c1-stock34 | c3-stock34 | 10/10 | not measured | - | whole text |
| c1-stock34 | c3-tier6g | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| c1-stock34 | c4-stock34 | 10/10 | not measured | - | whole text |
| c1-stock34 | c4-tier6g | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| c1-stock34 | c5-stock34 | 10/10 | not measured | - | whole text |
| c1-stock34 | c5-tier6g | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| c1-tier6g | c2-stock34 | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| c1-tier6g | c2-tier6g | 10/10 | not measured | - | whole text |
| c1-tier6g | c3-stock34 | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| c1-tier6g | c3-tier6g | 10/10 | not measured | - | whole text |
| c1-tier6g | c4-stock34 | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| c1-tier6g | c4-tier6g | 10/10 | not measured | - | whole text |
| c1-tier6g | c5-stock34 | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| c1-tier6g | c5-tier6g | 10/10 | not measured | - | whole text |
| c2-stock34 | c2-tier6g | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| c2-stock34 | c3-stock34 | 10/10 | not measured | - | whole text |
| c2-stock34 | c3-tier6g | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| c2-stock34 | c4-stock34 | 10/10 | not measured | - | whole text |
| c2-stock34 | c4-tier6g | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| c2-stock34 | c5-stock34 | 10/10 | not measured | - | whole text |
| c2-stock34 | c5-tier6g | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| c2-tier6g | c3-stock34 | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| c2-tier6g | c3-tier6g | 10/10 | not measured | - | whole text |
| c2-tier6g | c4-stock34 | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| c2-tier6g | c4-tier6g | 10/10 | not measured | - | whole text |
| c2-tier6g | c5-stock34 | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| c2-tier6g | c5-tier6g | 10/10 | not measured | - | whole text |
| c3-stock34 | c3-tier6g | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| c3-stock34 | c4-stock34 | 10/10 | not measured | - | whole text |
| c3-stock34 | c4-tier6g | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| c3-stock34 | c5-stock34 | 10/10 | not measured | - | whole text |
| c3-stock34 | c5-tier6g | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| c3-tier6g | c4-stock34 | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| c3-tier6g | c4-tier6g | 10/10 | not measured | - | whole text |
| c3-tier6g | c5-stock34 | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| c3-tier6g | c5-tier6g | 10/10 | not measured | - | whole text |
| c4-stock34 | c4-tier6g | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| c4-stock34 | c5-stock34 | 10/10 | not measured | - | whole text |
| c4-stock34 | c5-tier6g | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| c4-tier6g | c5-stock34 | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
| c4-tier6g | c5-tier6g | 10/10 | not measured | - | whole text |
| c5-stock34 | c5-tier6g | 2/10 | not measured | - | shared prefix (lengths differ; texts compared by token id) |
