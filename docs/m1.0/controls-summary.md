| arm | decode tok/s | post-pp16K tok/s | pp4k tok/s | pp16K tok/s | hit median / p10 | MiB up/token | wait / host ms per token | VRAM peak | free RAM min | peak WS / commit GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| coder-floor-c2400 | 7.1 | 7.5 | 52.8 | 122.5 | 57.9% / 41.0% | 402.8 | 0.92 / 95.71 | 15813 | 0.15 | 21.46 / 13.54 |
| coder-floor-ncmoe45 | 6.5 | 2.8 | 88.7 | 103.7 | - / - | - | - / - | 15897 | 0.21 | 21.58 / 12.85 |
| coder-full-c9400 | 13.0 | 6.4 | 97.8 | 181.2 | 91.0% / 80.0% | 96.6 | 0.87 / 42.11 | 15442 | 0.22 | 21.76 / 20.43 |
| dense27-floor-ngl40 | 2.2 | 2.2 | 415.4 | - | - / - | - | - / - | 15718 | 2.18 | 16.2 / 14.52 |
| dense27-full | 29.5 | 29.8 | 1234.2 | - | - / - | - | - / - | 13784 | 4.77 | 16.21 / 18.5 |
| gemma-floor-c2500 | 42.6 | 40.5 | 756.9 | - | 87.5% / 72.9% | 93.8 | 0.34 / 8.52 | 15086 | 8.05 | 14.69 / 8.61 |
| gemma-floor-ncmoe24 | 16.8 | 19.8 | 700.8 | - | - / - | - | - / - | 14802 | 8.06 | 14.73 / 8.33 |
| gemma-full-ncmoe0 | 79.4 | 79.7 | 3645.7 | - | - / - | - | - / - | 14439 | 7.77 | 15.09 / 16.44 |
| q36-floor-c2500 | 15.5 | 18.2 | 181.6 | - | 70.6% / 51.2% | 181.4 | 0.81 / 38.07 | 14563 | 0.23 | 21.17 / 8.61 |
| q36-floor-ncmoe32 | 13.1 | 23.8 | 270.2 | - | - / - | - | - / - | 15727 | 0.24 | 20.53 / 9.87 |
| q36-full-c9000 | 43.3 | 37.3 | 367.3 | - | 95.9% / 88.4% | 31.6 | 0.58 / 4.75 | 12652 | 0.30 | 21.36 / 14.99 |

| arm A | arm B | decode texts identical | top-3 probs identical | compared over |
|---|---|---|---|---|
| coder-floor-c2400 | coder-floor-ncmoe45 | 3/10 | 3/10 | whole text |
| coder-floor-c2400 | coder-full-c9400 | 3/10 | 3/10 | shared prefix (lengths differ; texts compared by token id) |
| coder-floor-c2400 | dense27-floor-ngl40 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| coder-floor-c2400 | dense27-full | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| coder-floor-c2400 | gemma-floor-c2500 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| coder-floor-c2400 | gemma-floor-ncmoe24 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| coder-floor-c2400 | gemma-full-ncmoe0 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| coder-floor-c2400 | q36-floor-c2500 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| coder-floor-c2400 | q36-floor-ncmoe32 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| coder-floor-c2400 | q36-full-c9000 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| coder-floor-ncmoe45 | coder-full-c9400 | 4/10 | 4/10 | shared prefix (lengths differ; texts compared by token id) |
| coder-floor-ncmoe45 | dense27-floor-ngl40 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| coder-floor-ncmoe45 | dense27-full | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| coder-floor-ncmoe45 | gemma-floor-c2500 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| coder-floor-ncmoe45 | gemma-floor-ncmoe24 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| coder-floor-ncmoe45 | gemma-full-ncmoe0 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| coder-floor-ncmoe45 | q36-floor-c2500 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| coder-floor-ncmoe45 | q36-floor-ncmoe32 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| coder-floor-ncmoe45 | q36-full-c9000 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| coder-full-c9400 | dense27-floor-ngl40 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| coder-full-c9400 | dense27-full | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| coder-full-c9400 | gemma-floor-c2500 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| coder-full-c9400 | gemma-floor-ncmoe24 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| coder-full-c9400 | gemma-full-ncmoe0 | 0/10 | 0/10 | whole text |
| coder-full-c9400 | q36-floor-c2500 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| coder-full-c9400 | q36-floor-ncmoe32 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| coder-full-c9400 | q36-full-c9000 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| dense27-floor-ngl40 | dense27-full | 2/10 | 2/10 | shared prefix (lengths differ; texts compared by token id) |
| dense27-floor-ngl40 | gemma-floor-c2500 | 0/10 | 0/10 | whole text |
| dense27-floor-ngl40 | gemma-floor-ncmoe24 | 0/10 | 0/10 | whole text |
| dense27-floor-ngl40 | gemma-full-ncmoe0 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| dense27-floor-ngl40 | q36-floor-c2500 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| dense27-floor-ngl40 | q36-floor-ncmoe32 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| dense27-floor-ngl40 | q36-full-c9000 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| dense27-full | gemma-floor-c2500 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| dense27-full | gemma-floor-ncmoe24 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| dense27-full | gemma-full-ncmoe0 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| dense27-full | q36-floor-c2500 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| dense27-full | q36-floor-ncmoe32 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| dense27-full | q36-full-c9000 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gemma-floor-c2500 | gemma-floor-ncmoe24 | 0/10 | 0/10 | whole text |
| gemma-floor-c2500 | gemma-full-ncmoe0 | 10/10 | 10/10 | shared prefix (lengths differ; texts compared by token id) |
| gemma-floor-c2500 | q36-floor-c2500 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gemma-floor-c2500 | q36-floor-ncmoe32 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gemma-floor-c2500 | q36-full-c9000 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gemma-floor-ncmoe24 | gemma-full-ncmoe0 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gemma-floor-ncmoe24 | q36-floor-c2500 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gemma-floor-ncmoe24 | q36-floor-ncmoe32 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gemma-floor-ncmoe24 | q36-full-c9000 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gemma-full-ncmoe0 | q36-floor-c2500 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gemma-full-ncmoe0 | q36-floor-ncmoe32 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gemma-full-ncmoe0 | q36-full-c9000 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| q36-floor-c2500 | q36-floor-ncmoe32 | 3/10 | 3/10 | shared prefix (lengths differ; texts compared by token id) |
| q36-floor-c2500 | q36-full-c9000 | 7/10 | 7/10 | shared prefix (lengths differ; texts compared by token id) |
| q36-floor-ncmoe32 | q36-full-c9000 | 3/10 | 3/10 | shared prefix (lengths differ; texts compared by token id) |
