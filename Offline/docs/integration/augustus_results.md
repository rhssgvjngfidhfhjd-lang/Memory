# AUGUSTUSMemory results

CSV source: `augustus_results.csv`

## Qwen/Qwen3-VL-4B-Instruct

Run-ID: `0913_augustus_rebuild`

| Benchmark | Baseline | F1 | EM | Judge | Cost-MB | Cost-QA | Lat-MB | Lat-QA | #Calls (MB+QA) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H2HMEM | AUGUSTUSMemory | 0.294461 | 0.077778 | 0.392361 | 0.005897 | 0.018263 | 71.3044 | 123.9463 | 214.40 |
| WorldMemArena | AUGUSTUSMemory | 0.229431 | 0.088636 | 0.663636 | 0.011163 | 0.011934 | 120.8718 | 99.4187 | 341.00 |
| Mem-Gallery | AUGUSTUSMemory | 0.478711 | 0.294545 | 0.694545 | 0.005398 | 0.010424 | 67.9297 | 115.2749 | 235.75 |

## openai/gpt-5-mini

Run-ID: `0913_augustus_gpt5mini_rebuild_v3`

| Benchmark | Baseline | F1 | EM | Judge | Cost-MB | Cost-QA | Lat-MB | Lat-QA | #Calls (MB+QA) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H2HMEM | AUGUSTUSMemory | 0.334892 | 0.094444 | 0.458333 | 0.037835 | 0.102417 | 616.2787 | 947.8339 | 215.20 |
| WorldMemArena | AUGUSTUSMemory | 0.228721 | 0.031818 | 0.802273 | 0.110695 | 0.091012 | 1768.6742 | 936.1457 | 469.50 |
| Mem-Gallery | AUGUSTUSMemory | 0.523843 | 0.272727 | 0.721818 | 0.038605 | 0.064245 | 709.9027 | 919.0474 | 245.50 |

`Cost-MB` and `Cost-QA` are USD/sample. `Lat-MB` and `Lat-QA` are estimated seconds/sample. The two model profiles use different pricing and latency coefficients, so cost and latency should be compared with the model column retained.
