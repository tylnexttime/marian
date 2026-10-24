| arm | decode tok/s: median / all tokens | post-pp16K tok/s: median / all tokens | pp4k tok/s | pp16K tok/s | hit median / p10 | MiB up/token | wait / host ms per token | VRAM peak | free RAM min | peak WS / commit GiB |
|---|---|---|---|---|---|---|---|---|---|---|
| gate11-gemma26-c2500-mmap | 8.1 / 8.1 | 12.0 / 10.6 | 122.6 | 199.9 | 83.3% / 66.2% | 117.3 | 0.61 / 59.92 | 15274 | 0.13 | 6.8 / 9.64 |
| gate11-gemma26-c2500-ring | 13.4 / 14.1 | 12.9 / 14.2 | 48.1 | 218.7 | 83.3% / 66.2% | 117.3 | 0.47 / 42.99 | 15264 | 0.21 | 7.17 / 9.72 |

Speed excludes requests that stopped within 8 tokens (counted here): none

| arm A | arm B | decode texts identical | top-3 probs identical | compared over |
|---|---|---|---|---|
| gate11-gemma26-c2500-mmap | gate11-gemma26-c2500-ring | 10/10 | 10/10 | whole text |
