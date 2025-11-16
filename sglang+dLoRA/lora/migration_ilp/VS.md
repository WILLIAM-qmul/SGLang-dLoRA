# request_rate=100.0


## num_prompts: 100

root@61934144cffb:/workspace/sglang/benchmark/lora# python lora_bench_multi.py --use-trace
Namespace(backend='sglang', base_url=None, host='0.0.0.0', port=30000, num_prompts=100, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, request_rate=100.0, base_only=False, output_file=None, seed=1, lora_assignment_strategy='sequential', worker_urls=['http://127.0.0.1:30001', 'http://127.0.0.1:30002'], use_trace=True, trace_name='azure_v2', trace_path='/workspace/datasets/maf2/', trace_start_time='0.0.0', trace_end_time='0.0.60', trace_interval_minutes=60, trace_arrival_distribution='gamma', trace_total_rate=4.0, trace_cv=1.0, trace_map_stride=1, trace_need_sort=False, inference_architecture='dlora', migration_type='period_mig', migration_interval=10, migration_req_threshold=16, max_loras_per_batch=8, max_running_requests=16)

#Input tokens: 40315
#Output tokens: 25466
[EngineManager] Initialized:
  - Instances: 2
  - LoRAs: 4
  - Exec Type: LORA
  - Migration Type: PERIOD_MIG
  - Initial LoRA Placement: {0: [1, 3, 2], 1: [2, 0]}
[EngineManager] Starting background migration loop...

[Benchmark] EngineManager Stats: {'num_instances': 2, 'num_loras': 4, 'instance_requests': [0, 0], 'lora_placement': {0: [1, 3, 2], 1: [2, 0]}, 'instance_exec_cost': {}, 'migration_type': 'PERIOD_MIG', 'exec_type': 'LORA'}
Starting initial single prompt test run...
Test request: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.76s/it]
Initial test run completed. Starting main benchmark run...
  0%|                                                                                                                                                                                                                             | 0/100 [00:00<?, ?it/s][18, 22, 8, 52]
last arrival: 28.47858551709426
  8%|█████████████████                                                                                                                                                                                                    | 8/100 [00:04<00:32,  2.79it/s][EngineManager] No request metadata, skip migration
 40%|████████████████████████████████████████████████████████████████████████████████████▊                                                                                                                               | 40/100 [00:15<00:16,  3.59it/s][EngineManager] No request metadata, skip migration
 72%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                           | 72/100 [00:24<00:08,  3.44it/s][EngineManager] No request metadata, skip migration
 98%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊    | 98/100 [00:34<00:01,  1.85it/s][EngineManager] No request metadata, skip migration
 99%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉  | 99/100 [00:38<00:01,  1.24s/it][EngineManager] No request metadata, skip migration
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:45<00:00,  2.20it/s]
[EngineManager] Stopping background migration loop...

============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    100.0     
Successful requests:                     100       
Benchmark duration (s):                  45.41     
Total input tokens:                      40315     
Total generated tokens:                  25466     
Total generated tokens (retokenized):    10592     
Request throughput (req/s):              2.20      
Input token throughput (tok/s):          887.83    
Output token throughput (tok/s):         560.82    
Total throughput (tok/s):                1448.66   
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   1974.12   
Median E2E Latency (ms):                 443.87    
---------------Time to First Token----------------
Mean TTFT (ms):                          1944.20   
Median TTFT (ms):                        443.89    
P99 TTFT (ms):                           15399.18  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          2.10      
Median TPOT (ms):                        -0.00     
P99 TPOT (ms):                           27.75     
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P99 ITL (ms):                            0.00      
==================================================
root@61934144cffb:/workspace/sglang/benchmark/lora# python lora_bench_multi.py --use-trace
Namespace(backend='sglang', base_url=None, host='0.0.0.0', port=30000, num_prompts=100, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, request_rate=100.0, base_only=False, output_file=None, seed=1, lora_assignment_strategy='sequential', worker_urls=['http://127.0.0.1:30001', 'http://127.0.0.1:30002'], use_trace=True, trace_name='azure_v2', trace_path='/workspace/datasets/maf2/', trace_start_time='0.0.0', trace_end_time='0.0.60', trace_interval_minutes=60, trace_arrival_distribution='gamma', trace_total_rate=4.0, trace_cv=1.0, trace_map_stride=1, trace_need_sort=False, inference_architecture='sglang', migration_type='period_mig', migration_interval=10, migration_req_threshold=16, max_loras_per_batch=8, max_running_requests=16)

#Input tokens: 40315
#Output tokens: 25466

[Benchmark] 静态 LoRA 路由 (sglang):
  lora0 -> http://127.0.0.1:30001/generate
  lora1 -> http://127.0.0.1:30002/generate
  lora2 -> http://127.0.0.1:30001/generate
  lora3 -> http://127.0.0.1:30002/generate
Starting initial single prompt test run...
Test request: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.75s/it]
Initial test run completed. Starting main benchmark run...
  0%|                                                                                                                                                                                                                             | 0/100 [00:00<?, ?it/s][18, 22, 8, 52]
last arrival: 28.47858551709426
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:43<00:00,  2.31it/s]

============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    100.0     
Successful requests:                     100       
Benchmark duration (s):                  43.38     
Total input tokens:                      40315     
Total generated tokens:                  25466     
Total generated tokens (retokenized):    13183     
Request throughput (req/s):              2.31      
Input token throughput (tok/s):          929.38    
Output token throughput (tok/s):         587.07    
Total throughput (tok/s):                1516.45   
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   2486.45   
Median E2E Latency (ms):                 457.89    
---------------Time to First Token----------------
Mean TTFT (ms):                          2451.06   
Median TTFT (ms):                        457.92    
P99 TTFT (ms):                           14833.07  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          2.13      
Median TPOT (ms):                        -0.00     
P99 TPOT (ms):                           28.52     
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P99 ITL (ms):                            0.00      
==================================================


## num_prompts: 1000

root@61934144cffb:/workspace/sglang/benchmark/lora# python lora_bench_multi.py --use-trace
Namespace(backend='sglang', base_url=None, host='0.0.0.0', port=30000, num_prompts=1000, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, request_rate=100.0, base_only=False, output_file=None, seed=1, lora_assignment_strategy='sequential', worker_urls=['http://127.0.0.1:30001', 'http://127.0.0.1:30002'], use_trace=True, trace_name='azure_v2', trace_path='/workspace/datasets/maf2/', trace_start_time='0.0.0', trace_end_time='0.0.60', trace_interval_minutes=60, trace_arrival_distribution='gamma', trace_total_rate=4.0, trace_cv=1.0, trace_map_stride=1, trace_need_sort=False, inference_architecture='sglang', migration_type='period_mig', migration_interval=10, migration_req_threshold=16, max_loras_per_batch=8, max_running_requests=16)

#Input tokens: 360334
#Output tokens: 223999

[Benchmark] 静态 LoRA 路由 (sglang):
  lora0 -> http://127.0.0.1:30001/generate
  lora1 -> http://127.0.0.1:30002/generate
  lora2 -> http://127.0.0.1:30001/generate
  lora3 -> http://127.0.0.1:30002/generate
Starting initial single prompt test run...
Test request: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.68s/it]
Initial test run completed. Starting main benchmark run...
  0%|                                                                                                                                                                                                                            | 0/1000 [00:00<?, ?it/s][189, 253, 89, 469]
last arrival: 250.5046187369351
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [04:25<00:00,  3.77it/s]

============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    100.0     
Successful requests:                     996       
Benchmark duration (s):                  265.19    
Total input tokens:                      342460    
Total generated tokens:                  218257    
Total generated tokens (retokenized):    98570     
Request throughput (req/s):              3.76      
Input token throughput (tok/s):          1291.36   
Output token throughput (tok/s):         823.01    
Total throughput (tok/s):                2114.37   
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   1945.91   
Median E2E Latency (ms):                 512.41    
---------------Time to First Token----------------
Mean TTFT (ms):                          1921.74   
Median TTFT (ms):                        512.43    
P99 TTFT (ms):                           14724.47  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          2.02      
Median TPOT (ms):                        -0.00     
P99 TPOT (ms):                           44.25     
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P99 ITL (ms):                            0.00      
==================================================
root@61934144cffb:/workspace/sglang/benchmark/lora# python lora_bench_multi.py --use-trace
Namespace(backend='sglang', base_url=None, host='0.0.0.0', port=30000, num_prompts=1000, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, request_rate=100.0, base_only=False, output_file=None, seed=1, lora_assignment_strategy='sequential', worker_urls=['http://127.0.0.1:30001', 'http://127.0.0.1:30002'], use_trace=True, trace_name='azure_v2', trace_path='/workspace/datasets/maf2/', trace_start_time='0.0.0', trace_end_time='0.0.60', trace_interval_minutes=60, trace_arrival_distribution='gamma', trace_total_rate=4.0, trace_cv=1.0, trace_map_stride=1, trace_need_sort=False, inference_architecture='dlora', migration_type='period_mig', migration_interval=10, migration_req_threshold=16, max_loras_per_batch=8, max_running_requests=16)

#Input tokens: 360334
#Output tokens: 223999
[EngineManager] Initialized:
  - Instances: 2
  - LoRAs: 4
  - Exec Type: LORA
  - Migration Type: PERIOD_MIG
  - Initial LoRA Placement: {0: [1, 3, 2], 1: [2, 0]}
[EngineManager] Starting background migration loop...

[Benchmark] EngineManager Stats: {'num_instances': 2, 'num_loras': 4, 'instance_requests': [0, 0], 'lora_placement': {0: [1, 3, 2], 1: [2, 0]}, 'instance_exec_cost': {}, 'migration_type': 'PERIOD_MIG', 'exec_type': 'LORA'}
Starting initial single prompt test run...
Test request: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.74s/it]
Initial test run completed. Starting main benchmark run...
  0%|                                                                                                                                                                                                                            | 0/1000 [00:00<?, ?it/s][189, 253, 89, 469]
last arrival: 250.5046187369351
  1%|█▎                                                                                                                                                                                                                  | 6/1000 [00:05<09:23,  1.76it/s][EngineManager] No request metadata, skip migration
  4%|███████▊                                                                                                                                                                                                           | 37/1000 [00:15<05:46,  2.78it/s][EngineManager] No request metadata, skip migration
  7%|███████████████▌                                                                                                                                                                                                   | 74/1000 [00:25<03:07,  4.93it/s][EngineManager] No request metadata, skip migration
 11%|██████████████████████▉                                                                                                                                                                                           | 109/1000 [00:34<02:12,  6.73it/s][EngineManager] No request metadata, skip migration
 15%|███████████████████████████████▌                                                                                                                                                                                  | 150/1000 [00:45<01:55,  7.37it/s][EngineManager] No request metadata, skip migration
 20%|█████████████████████████████████████████▊                                                                                                                                                                        | 199/1000 [00:55<02:06,  6.35it/s][EngineManager] No request metadata, skip migration
 24%|█████████████████████████████████████████████████▊                                                                                                                                                                | 237/1000 [01:04<03:16,  3.88it/s][EngineManager] No request metadata, skip migration
 28%|██████████████████████████████████████████████████████████▌                                                                                                                                                       | 279/1000 [01:15<03:23,  3.55it/s][EngineManager] No request metadata, skip migration
 31%|████████████████████████████████████████████████████████████████▍                                                                                                                                                 | 307/1000 [01:25<04:31,  2.55it/s][EngineManager] No request metadata, skip migration
 34%|████████████████████████████████████████████████████████████████████████▏                                                                                                                                         | 344/1000 [01:35<03:29,  3.13it/s][EngineManager] No request metadata, skip migration
 38%|████████████████████████████████████████████████████████████████████████████████                                                                                                                                  | 381/1000 [01:44<03:33,  2.90it/s][EngineManager] No request metadata, skip migration
 43%|█████████████████████████████████████████████████████████████████████████████████████████▋                                                                                                                        | 427/1000 [01:54<01:45,  5.44it/s][EngineManager] No request metadata, skip migration
 46%|████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                                 | 460/1000 [02:05<03:07,  2.88it/s][EngineManager] No request metadata, skip migration
 51%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                                                                      | 514/1000 [02:14<02:04,  3.91it/s][EngineManager] No request metadata, skip migration
 56%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                             | 555/1000 [02:25<02:09,  3.42it/s][EngineManager] No request metadata, skip migration
 59%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                                                                                      | 587/1000 [02:34<02:09,  3.18it/s][EngineManager] No request metadata, skip migration
 63%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                                              | 627/1000 [02:45<01:41,  3.67it/s][EngineManager] No request metadata, skip migration
 66%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                       | 660/1000 [02:55<02:15,  2.51it/s][EngineManager] No request metadata, skip migration
 70%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                              | 705/1000 [03:05<01:33,  3.16it/s][EngineManager] No request metadata, skip migration
 75%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                                     | 746/1000 [03:14<01:23,  3.03it/s][EngineManager] No request metadata, skip migration
 80%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                          | 796/1000 [03:24<00:32,  6.29it/s][EngineManager] No request metadata, skip migration
 83%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                   | 833/1000 [03:34<00:57,  2.91it/s][EngineManager] No request metadata, skip migration
 87%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                           | 872/1000 [03:45<00:33,  3.80it/s][EngineManager] No request metadata, skip migration
 91%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                  | 913/1000 [03:55<00:45,  1.92it/s][EngineManager] No request metadata, skip migration
 96%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊         | 956/1000 [04:05<00:09,  4.82it/s][EngineManager] No request metadata, skip migration
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉ | 995/1000 [04:15<00:01,  2.54it/s][EngineManager] No request metadata, skip migration
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [04:22<00:00,  3.81it/s]
[EngineManager] Stopping background migration loop...

============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    100.0     
Successful requests:                     996       
Benchmark duration (s):                  262.27    
Total input tokens:                      342460    
Total generated tokens:                  218257    
Total generated tokens (retokenized):    98099     
Request throughput (req/s):              3.80      
Input token throughput (tok/s):          1305.77   
Output token throughput (tok/s):         832.19    
Total throughput (tok/s):                2137.96   
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   1958.60   
Median E2E Latency (ms):                 535.31    
---------------Time to First Token----------------
Mean TTFT (ms):                          1935.11   
Median TTFT (ms):                        535.33    
P99 TTFT (ms):                           14670.10  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          1.72      
Median TPOT (ms):                        -0.00     
P99 TPOT (ms):                           26.82     
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P99 ITL (ms):                            0.00      
==================================================
root@61934144cffb:/workspace/sglang/benchmark/lora# 