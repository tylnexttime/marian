| arm | decode tok/s | post-pp16K tok/s | pp4k tok/s | pp16K tok/s | hit median / p10 | MiB up/token | wait / host ms per token | VRAM peak | free RAM min | peak WS / commit GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| f1-c2400 | 4.9 | 3.2 | 28.1 | 38.3 | 63.7% / 45.0% | 242.8 | 1.42 / 131.64 | 15488 | 0.03 | 23.07 / 13.25 |
| f1-stock45 | 7.7 | 5.6 | 27.0 | 30.5 | - / - | - | - / - | 15623 | 0.08 | 22.66 / 13.01 |
| f2-c2400 | 5.9 | 3.6 | 33.9 | 42.4 | 63.7% / 45.0% | 242.8 | 0.87 / 103.85 | 15766 | 0.11 | 23.24 / 13.26 |
| f2-stock45 | 7.5 | 5.8 | 27.9 | 27.7 | - / - | - | - / - | 15881 | 0.10 | 22.67 / 13.05 |
| r1-c4000 | 7.1 | 3.2 | 34.6 | 50.2 | 74.8% / 56.7% | 175.6 | 0.76 / 83.42 | 9058 | 0.17 | 23.03 / 14.85 |
| r1-c6500 | 6.9 | 3.7 | 45.1 | 80.4 | 84.4% / 69.8% | 112.7 | 0.82 / 78.31 | 11562 | 0.18 | 22.54 / 17.29 |
| r1-c9400 | 7.9 | 4.0 | 54.1 | 107.2 | 90.8% / 80.4% | 69.3 | 0.79 / 72.98 | 14462 | 0.06 | 22.5 / 20.13 |
| r1-stock34 | 10.6 | 5.1 | 111.6 | 152.0 | - / - | - | - / - | 15200 | 0.16 | 22.28 / 20.83 |
| r2-c4000 | 6.9 | 3.3 | 33.6 | 51.5 | 74.8% / 56.7% | 175.6 | 0.85 / 87.71 | 9058 | 0.16 | 22.83 / 14.85 |
| r2-c6500 | 7.1 | 3.6 | 40.9 | 86.1 | 84.4% / 69.8% | 112.7 | 0.83 / 80.07 | 11562 | 0.15 | 22.45 / 17.29 |
| r2-c9400 | 7.3 | 3.5 | 51.6 | 107.9 | 90.8% / 80.4% | 69.3 | 0.79 / 70.73 | 14462 | 0.11 | 22.43 / 20.13 |
| r2-stock34 | 10.9 | 4.7 | 113.2 | 147.9 | - / - | - | - / - | 15200 | 0.18 | 22.26 / 20.83 |

| arm A | arm B | decode texts identical | top-3 probs identical | compared over |
|---|---|---|---|---|
| f1-c2400 | f1-stock45 | 4/10 | 4/10 | shared prefix (lengths differ; texts compared by token id) |
| f1-c2400 | f2-c2400 | 10/10 | 10/10 | whole text |
| f1-c2400 | f2-stock45 | 4/10 | 4/10 | shared prefix (lengths differ; texts compared by token id) |
| f1-c2400 | r1-c4000 | 5/10 | 5/10 | shared prefix (lengths differ; texts compared by token id) |
| f1-c2400 | r1-c6500 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| f1-c2400 | r1-c9400 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| f1-c2400 | r1-stock34 | 3/10 | 3/10 | shared prefix (lengths differ; texts compared by token id) |
| f1-c2400 | r2-c4000 | 5/10 | 5/10 | shared prefix (lengths differ; texts compared by token id) |
| f1-c2400 | r2-c6500 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| f1-c2400 | r2-c9400 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| f1-c2400 | r2-stock34 | 3/10 | 3/10 | shared prefix (lengths differ; texts compared by token id) |
| f1-stock45 | f2-c2400 | 4/10 | 4/10 | shared prefix (lengths differ; texts compared by token id) |
| f1-stock45 | f2-stock45 | 10/10 | 10/10 | whole text |
| f1-stock45 | r1-c4000 | 4/10 | 4/10 | shared prefix (lengths differ; texts compared by token id) |
| f1-stock45 | r1-c6500 | 4/10 | 4/10 | shared prefix (lengths differ; texts compared by token id) |
| f1-stock45 | r1-c9400 | 4/10 | 4/10 | shared prefix (lengths differ; texts compared by token id) |
| f1-stock45 | r1-stock34 | 6/10 | 6/10 | shared prefix (lengths differ; texts compared by token id) |
| f1-stock45 | r2-c4000 | 4/10 | 4/10 | shared prefix (lengths differ; texts compared by token id) |
| f1-stock45 | r2-c6500 | 4/10 | 4/10 | shared prefix (lengths differ; texts compared by token id) |
| f1-stock45 | r2-c9400 | 4/10 | 4/10 | shared prefix (lengths differ; texts compared by token id) |
| f1-stock45 | r2-stock34 | 6/10 | 6/10 | shared prefix (lengths differ; texts compared by token id) |
| f2-c2400 | f2-stock45 | 4/10 | 4/10 | shared prefix (lengths differ; texts compared by token id) |
| f2-c2400 | r1-c4000 | 5/10 | 5/10 | shared prefix (lengths differ; texts compared by token id) |
| f2-c2400 | r1-c6500 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| f2-c2400 | r1-c9400 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| f2-c2400 | r1-stock34 | 3/10 | 3/10 | shared prefix (lengths differ; texts compared by token id) |
| f2-c2400 | r2-c4000 | 5/10 | 5/10 | shared prefix (lengths differ; texts compared by token id) |
| f2-c2400 | r2-c6500 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| f2-c2400 | r2-c9400 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| f2-c2400 | r2-stock34 | 3/10 | 3/10 | shared prefix (lengths differ; texts compared by token id) |
| f2-stock45 | r1-c4000 | 4/10 | 4/10 | shared prefix (lengths differ; texts compared by token id) |
| f2-stock45 | r1-c6500 | 4/10 | 4/10 | shared prefix (lengths differ; texts compared by token id) |
| f2-stock45 | r1-c9400 | 4/10 | 4/10 | shared prefix (lengths differ; texts compared by token id) |
| f2-stock45 | r1-stock34 | 6/10 | 6/10 | shared prefix (lengths differ; texts compared by token id) |
| f2-stock45 | r2-c4000 | 4/10 | 4/10 | shared prefix (lengths differ; texts compared by token id) |
| f2-stock45 | r2-c6500 | 4/10 | 4/10 | shared prefix (lengths differ; texts compared by token id) |
| f2-stock45 | r2-c9400 | 4/10 | 4/10 | shared prefix (lengths differ; texts compared by token id) |
| f2-stock45 | r2-stock34 | 6/10 | 6/10 | shared prefix (lengths differ; texts compared by token id) |
| r1-c4000 | r1-c6500 | 7/10 | 7/10 | shared prefix (lengths differ; texts compared by token id) |
| r1-c4000 | r1-c9400 | 7/10 | 7/10 | shared prefix (lengths differ; texts compared by token id) |
| r1-c4000 | r1-stock34 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| r1-c4000 | r2-c4000 | 10/10 | 10/10 | whole text |
| r1-c4000 | r2-c6500 | 7/10 | 7/10 | shared prefix (lengths differ; texts compared by token id) |
| r1-c4000 | r2-c9400 | 7/10 | 7/10 | shared prefix (lengths differ; texts compared by token id) |
| r1-c4000 | r2-stock34 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| r1-c6500 | r1-c9400 | 10/10 | 10/10 | whole text |
| r1-c6500 | r1-stock34 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| r1-c6500 | r2-c4000 | 7/10 | 7/10 | shared prefix (lengths differ; texts compared by token id) |
| r1-c6500 | r2-c6500 | 10/10 | 10/10 | whole text |
| r1-c6500 | r2-c9400 | 10/10 | 10/10 | whole text |
| r1-c6500 | r2-stock34 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| r1-c9400 | r1-stock34 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| r1-c9400 | r2-c4000 | 7/10 | 7/10 | shared prefix (lengths differ; texts compared by token id) |
| r1-c9400 | r2-c6500 | 10/10 | 10/10 | whole text |
| r1-c9400 | r2-c9400 | 10/10 | 10/10 | whole text |
| r1-c9400 | r2-stock34 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| r1-stock34 | r2-c4000 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| r1-stock34 | r2-c6500 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| r1-stock34 | r2-c9400 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| r1-stock34 | r2-stock34 | 10/10 | 10/10 | whole text |
| r2-c4000 | r2-c6500 | 7/10 | 7/10 | shared prefix (lengths differ; texts compared by token id) |
| r2-c4000 | r2-c9400 | 7/10 | 7/10 | shared prefix (lengths differ; texts compared by token id) |
| r2-c4000 | r2-stock34 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| r2-c6500 | r2-c9400 | 10/10 | 10/10 | whole text |
| r2-c6500 | r2-stock34 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| r2-c9400 | r2-stock34 | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
