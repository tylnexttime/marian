| arm | decode tok/s | post-pp16K tok/s | pp4k tok/s | pp16K tok/s | hit median / p10 | MiB up/token | wait / host ms per token | VRAM peak | free RAM min | peak WS / commit GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| p3-r1-c9400 | 7.1 | 3.5 | 53.3 | 119.3 | 90.8% / 80.4% | 69.3 | 0.94 / 75.37 | 14991 | 0.23 | 22.07 / 20.14 |
| p3-r1-c9400-ring | 11.7 | 7.6 | 45.0 | 105.0 | 90.8% / 80.4% | 69.3 | 0.77 / 48.01 | 15021 | 0.14 | 21.78 / 20.26 |
| p3-r1-c9400-tier6g | 14.2 | 10.9 | 25.5 | 34.0 | 90.8% / 80.4% | 69.3 | 0.77 / 39.11 | 15021 | 0.11 | 21.48 / 26.27 |
| p3-r1-stock34 | 8.2 | 4.1 | 104.1 | 145.2 | - / - | - | - / - | 15863 | 0.16 | 21.81 / 20.84 |
| p3-r2-c9400 | 6.1 | 3.3 | 47.2 | 112.0 | 90.8% / 80.4% | 69.3 | 1.06 / 83.91 | 14990 | 0.21 | 21.57 / 20.13 |
| p3-r2-c9400-ring | 11.5 | 8.4 | 41.0 | 109.9 | 90.8% / 80.4% | 69.3 | 0.76 / 47.91 | 15020 | 0.13 | 21.75 / 20.25 |
| p3-r2-c9400-tier6g | 13.8 | 10.3 | 23.0 | 35.8 | 90.8% / 80.4% | 69.3 | 0.76 / 39.00 | 15020 | 0.10 | 20.73 / 26.27 |
| p3-r2-stock34 | 10.4 | 3.7 | 96.1 | 136.4 | - / - | - | - / - | 15728 | 0.11 | 21.77 / 20.83 |

| arm A | arm B | decode texts identical | top-3 probs identical | compared over |
|---|---|---|---|---|
| p3-r1-c9400 | p3-r1-c9400-ring | 10/10 | 10/10 | whole text |
| p3-r1-c9400 | p3-r1-c9400-tier6g | 10/10 | 10/10 | whole text |
| p3-r1-c9400 | p3-r1-stock34 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| p3-r1-c9400 | p3-r2-c9400 | 10/10 | 10/10 | whole text |
| p3-r1-c9400 | p3-r2-c9400-ring | 10/10 | 10/10 | whole text |
| p3-r1-c9400 | p3-r2-c9400-tier6g | 10/10 | 10/10 | whole text |
| p3-r1-c9400 | p3-r2-stock34 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| p3-r1-c9400-ring | p3-r1-c9400-tier6g | 10/10 | 10/10 | whole text |
| p3-r1-c9400-ring | p3-r1-stock34 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| p3-r1-c9400-ring | p3-r2-c9400 | 10/10 | 10/10 | whole text |
| p3-r1-c9400-ring | p3-r2-c9400-ring | 10/10 | 10/10 | whole text |
| p3-r1-c9400-ring | p3-r2-c9400-tier6g | 10/10 | 10/10 | whole text |
| p3-r1-c9400-ring | p3-r2-stock34 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| p3-r1-c9400-tier6g | p3-r1-stock34 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| p3-r1-c9400-tier6g | p3-r2-c9400 | 10/10 | 10/10 | whole text |
| p3-r1-c9400-tier6g | p3-r2-c9400-ring | 10/10 | 10/10 | whole text |
| p3-r1-c9400-tier6g | p3-r2-c9400-tier6g | 10/10 | 10/10 | whole text |
| p3-r1-c9400-tier6g | p3-r2-stock34 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| p3-r1-stock34 | p3-r2-c9400 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| p3-r1-stock34 | p3-r2-c9400-ring | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| p3-r1-stock34 | p3-r2-c9400-tier6g | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| p3-r1-stock34 | p3-r2-stock34 | 10/10 | 10/10 | whole text |
| p3-r2-c9400 | p3-r2-c9400-ring | 10/10 | 10/10 | whole text |
| p3-r2-c9400 | p3-r2-c9400-tier6g | 10/10 | 10/10 | whole text |
| p3-r2-c9400 | p3-r2-stock34 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| p3-r2-c9400-ring | p3-r2-c9400-tier6g | 10/10 | 10/10 | whole text |
| p3-r2-c9400-ring | p3-r2-stock34 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| p3-r2-c9400-tier6g | p3-r2-stock34 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
