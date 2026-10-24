| arm | decode tok/s: median / all tokens | post-pp16K tok/s: median / all tokens | pp4k tok/s | pp16K tok/s | hit median / p10 | MiB up/token | wait / host ms per token | VRAM peak | free RAM min | peak WS / commit GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| gate-bolt-e4b-full | 79.7 / 79.7 | 68.3 / 69.5 | 5488.8 | 5212.8 | - / - | - | - / - | 12703 | 2.99 | 4.38 / 5.39 |
| gate-e4b-qat-full | 89.7 / 89.6 | 86.0 / 85.9 | 6108.7 | 5643.6 | - / - | - | - / - | 12377 | 3.30 | 4.01 / 5.02 |
| gate-gemma26-c2500-desk25 | 8.0 / 7.5 | 7.4 / 7.4 | 97.8 | 169.3 | 87.5% / 72.9% | 93.8 | 0.65 / 65.87 | 15154 | 0.25 | 6.38 / 9.64 |
| gate-gemma26-iq3xxs-c2500 | 12.7 / 13.1 | 11.3 / 11.7 | 146.3 | 350.2 | 91.7% / 78.8% | 58.2 | 0.58 / 32.14 | 14651 | 0.12 | 6.42 / 9.1 |
| gate-gemma26-iq3xxs-ncmoe18 | 26.3 / 25.7 | 27.0 / 28.1 | 775.4 | 756.1 | - / - | - | - / - | 15833 | 0.13 | 6.5 / 10.25 |
| gate-gemma26-ncmoe24-desk25 | 15.0 / 14.9 | 15.7 / 15.2 | 129.1 | 322.5 | - / - | - | - / - | 14886 | 0.06 | 8.55 / 9.38 |
| gate-gemma26-ncmoe30-desk25 | 10.0 / 10.2 | 10.1 / 10.3 | 21.5 | 20.3 | - / - | - | - / - | 12791 | 0.07 | 7.05 / 7.2 |
| gate-groot-c2500 | 5.0 / 4.8 | 6.1 / 6.2 | 149.6 | 246.8 | 75.6% / 60.6% | 150.9 | 1.27 / 124.98 | 13633 | 0.17 | 7.06 / 8.25 |
| gate-groot-ncmoe28 | 19.2 / 18.2 | 20.6 / 21.3 | 267.3 | 447.9 | - / - | - | - / - | 13985 | 0.20 | 6.59 / 8.6 |
| gate-qwen35-9b-full | 65.4 / 65.2 | 65.6 / 65.6 | 3838.2 | 3829.7 | - / - | - | - / - | 15129 | 0.38 | 6.4 / 9.58 |

Speed excludes requests that stopped within 8 tokens (counted here): none

| arm A | arm B | decode texts identical | top-3 probs identical | compared over |
|---|---|---|---|---|
| gate-bolt-e4b-full | gate-e4b-qat-full | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gate-bolt-e4b-full | gate-gemma26-c2500-desk25 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gate-bolt-e4b-full | gate-gemma26-iq3xxs-c2500 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gate-bolt-e4b-full | gate-gemma26-iq3xxs-ncmoe18 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gate-bolt-e4b-full | gate-gemma26-ncmoe24-desk25 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gate-bolt-e4b-full | gate-gemma26-ncmoe30-desk25 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gate-bolt-e4b-full | gate-groot-c2500 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gate-bolt-e4b-full | gate-groot-ncmoe28 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gate-bolt-e4b-full | gate-qwen35-9b-full | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gate-e4b-qat-full | gate-gemma26-c2500-desk25 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gate-e4b-qat-full | gate-gemma26-iq3xxs-c2500 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gate-e4b-qat-full | gate-gemma26-iq3xxs-ncmoe18 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gate-e4b-qat-full | gate-gemma26-ncmoe24-desk25 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gate-e4b-qat-full | gate-gemma26-ncmoe30-desk25 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gate-e4b-qat-full | gate-groot-c2500 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gate-e4b-qat-full | gate-groot-ncmoe28 | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gate-e4b-qat-full | gate-qwen35-9b-full | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gate-gemma26-c2500-desk25 | gate-gemma26-iq3xxs-c2500 | 0/10 | 0/10 | whole text |
| gate-gemma26-c2500-desk25 | gate-gemma26-iq3xxs-ncmoe18 | 0/10 | 0/10 | whole text |
| gate-gemma26-c2500-desk25 | gate-gemma26-ncmoe24-desk25 | 0/10 | 0/10 | whole text |
| gate-gemma26-c2500-desk25 | gate-gemma26-ncmoe30-desk25 | 0/10 | 0/10 | whole text |
| gate-gemma26-c2500-desk25 | gate-groot-c2500 | 0/10 | 0/10 | whole text |
| gate-gemma26-c2500-desk25 | gate-groot-ncmoe28 | 0/10 | 0/10 | whole text |
| gate-gemma26-c2500-desk25 | gate-qwen35-9b-full | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gate-gemma26-iq3xxs-c2500 | gate-gemma26-iq3xxs-ncmoe18 | 0/10 | 0/10 | whole text |
| gate-gemma26-iq3xxs-c2500 | gate-gemma26-ncmoe24-desk25 | 0/10 | 0/10 | whole text |
| gate-gemma26-iq3xxs-c2500 | gate-gemma26-ncmoe30-desk25 | 0/10 | 0/10 | whole text |
| gate-gemma26-iq3xxs-c2500 | gate-groot-c2500 | 0/10 | 0/10 | whole text |
| gate-gemma26-iq3xxs-c2500 | gate-groot-ncmoe28 | 0/10 | 0/10 | whole text |
| gate-gemma26-iq3xxs-c2500 | gate-qwen35-9b-full | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gate-gemma26-iq3xxs-ncmoe18 | gate-gemma26-ncmoe24-desk25 | 0/10 | 0/10 | whole text |
| gate-gemma26-iq3xxs-ncmoe18 | gate-gemma26-ncmoe30-desk25 | 0/10 | 0/10 | whole text |
| gate-gemma26-iq3xxs-ncmoe18 | gate-groot-c2500 | 0/10 | 0/10 | whole text |
| gate-gemma26-iq3xxs-ncmoe18 | gate-groot-ncmoe28 | 0/10 | 0/10 | whole text |
| gate-gemma26-iq3xxs-ncmoe18 | gate-qwen35-9b-full | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gate-gemma26-ncmoe24-desk25 | gate-gemma26-ncmoe30-desk25 | 4/10 | 4/10 | whole text |
| gate-gemma26-ncmoe24-desk25 | gate-groot-c2500 | 0/10 | 0/10 | whole text |
| gate-gemma26-ncmoe24-desk25 | gate-groot-ncmoe28 | 0/10 | 0/10 | whole text |
| gate-gemma26-ncmoe24-desk25 | gate-qwen35-9b-full | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gate-gemma26-ncmoe30-desk25 | gate-groot-c2500 | 0/10 | 0/10 | whole text |
| gate-gemma26-ncmoe30-desk25 | gate-groot-ncmoe28 | 0/10 | 0/10 | whole text |
| gate-gemma26-ncmoe30-desk25 | gate-qwen35-9b-full | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gate-groot-c2500 | gate-groot-ncmoe28 | 2/10 | 2/10 | whole text |
| gate-groot-c2500 | gate-qwen35-9b-full | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
| gate-groot-ncmoe28 | gate-qwen35-9b-full | 0/10 | 0/10 | shared prefix (lengths differ; texts compared by token id) |
