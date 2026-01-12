# request rate = 2

root@61934144cffb:/workspace/sglang/benchmark/lora/migration_ilp# python auto_bench.py

Running: python3 /workspace/sglang/benchmark/lora/lora_bench_multi.py --num-prompts 1000 --request-rate 2 --inference-architecture sglang --output-file /workspace/sglang/benchmark/lora/migration_ilp/sglang/sglang_rate2.jsonl --use-trace
Namespace(backend='sglang', base_url=None, host='0.0.0.0', port=30000, num_prompts=1000, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, request_rate=2.0, base_only=False, output_file='/workspace/sglang/benchmark/lora/migration_ilp/sglang/sglang_rate2.jsonl', seed=1, lora_assignment_strategy='sequential', worker_urls=['http://127.0.0.1:30001', 'http://127.0.0.1:30002'], use_trace=True, trace_name='azure_v2', trace_path='/workspace/datasets/maf2/', trace_start_time='0.0.0', trace_end_time='0.0.60', trace_interval_minutes=60, trace_arrival_distribution='gamma', trace_total_rate=4.0, trace_cv=1.0, trace_map_stride=1, trace_need_sort=False, inference_architecture='sglang', migration_type='period_mig', migration_interval=10, migration_req_threshold=16, max_loras_per_batch=8, max_running_requests=16)

#Input tokens: 360334
#Output tokens: 223999

[Benchmark] 静态 LoRA 路由 (sglang):
  lora0 -> http://127.0.0.1:30001/generate
  lora1 -> http://127.0.0.1:30002/generate
  lora2 -> http://127.0.0.1:30001/generate
  lora3 -> http://127.0.0.1:30002/generate
Starting initial single prompt test run...
Test request: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.81s/it]
Initial test run completed. Starting main benchmark run...
  0%|                                                                                                                                                                                                                            | 0/1000 [00:00<?, ?it/s][189, 253, 89, 469]
last arrival: 250.5046187369351
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [04:22<00:00,  3.81it/s]

============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    2.0       
Successful requests:                     996       
Benchmark duration (s):                  262.38    
Total input tokens:                      342460    
Total generated tokens:                  218257    
Total generated tokens (retokenized):    100838    
Request throughput (req/s):              3.80      
Input token throughput (tok/s):          1305.20   
Output token throughput (tok/s):         831.83    
Total throughput (tok/s):                2137.04   
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   1979.33   
Median E2E Latency (ms):                 576.03    
---------------Time to First Token----------------
Mean TTFT (ms):                          1955.72   
Median TTFT (ms):                        576.10    
P99 TTFT (ms):                           13317.90  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          1.77      
Median TPOT (ms):                        -0.00     
P99 TPOT (ms):                           27.26     
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P99 ITL (ms):                            0.00      
==================================================


# request rate = 4

Running: python3 /workspace/sglang/benchmark/lora/lora_bench_multi.py --num-prompts 1000 --request-rate 4 --inference-architecture sglang --output-file /workspace/sglang/benchmark/lora/migration_ilp/sglang/sglang_rate4.jsonl --use-trace
Namespace(backend='sglang', base_url=None, host='0.0.0.0', port=30000, num_prompts=1000, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, request_rate=4.0, base_only=False, output_file='/workspace/sglang/benchmark/lora/migration_ilp/sglang/sglang_rate4.jsonl', seed=1, lora_assignment_strategy='sequential', worker_urls=['http://127.0.0.1:30001', 'http://127.0.0.1:30002'], use_trace=True, trace_name='azure_v2', trace_path='/workspace/datasets/maf2/', trace_start_time='0.0.0', trace_end_time='0.0.60', trace_interval_minutes=60, trace_arrival_distribution='gamma', trace_total_rate=4.0, trace_cv=1.0, trace_map_stride=1, trace_need_sort=False, inference_architecture='sglang', migration_type='period_mig', migration_interval=10, migration_req_threshold=16, max_loras_per_batch=8, max_running_requests=16)

#Input tokens: 360334
#Output tokens: 223999

[Benchmark] 静态 LoRA 路由 (sglang):
  lora0 -> http://127.0.0.1:30001/generate
  lora1 -> http://127.0.0.1:30002/generate
  lora2 -> http://127.0.0.1:30001/generate
  lora3 -> http://127.0.0.1:30002/generate
Starting initial single prompt test run...
Test request: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 43.37it/s]
Initial test run completed. Starting main benchmark run...
  0%|                                                                                                                                                                                                                            | 0/1000 [00:00<?, ?it/s][189, 253, 89, 469]
last arrival: 250.5046187369351
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [04:26<00:00,  3.75it/s]

============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    4.0       
Successful requests:                     996       
Benchmark duration (s):                  266.99    
Total input tokens:                      342460    
Total generated tokens:                  218257    
Total generated tokens (retokenized):    104559    
Request throughput (req/s):              3.73      
Input token throughput (tok/s):          1282.66   
Output token throughput (tok/s):         817.47    
Total throughput (tok/s):                2100.12   
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   2071.52   
Median E2E Latency (ms):                 594.76    
---------------Time to First Token----------------
Mean TTFT (ms):                          2048.41   
Median TTFT (ms):                        594.79    
P99 TTFT (ms):                           15249.17  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          1.77      
Median TPOT (ms):                        -0.00     
P99 TPOT (ms):                           26.12     
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P99 ITL (ms):                            0.00      
==================================================


# request rate = 8

Running: python3 /workspace/sglang/benchmark/lora/lora_bench_multi.py --num-prompts 1000 --request-rate 8 --inference-architecture sglang --output-file /workspace/sglang/benchmark/lora/migration_ilp/sglang/sglang_rate8.jsonl --use-trace
Namespace(backend='sglang', base_url=None, host='0.0.0.0', port=30000, num_prompts=1000, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, request_rate=8.0, base_only=False, output_file='/workspace/sglang/benchmark/lora/migration_ilp/sglang/sglang_rate8.jsonl', seed=1, lora_assignment_strategy='sequential', worker_urls=['http://127.0.0.1:30001', 'http://127.0.0.1:30002'], use_trace=True, trace_name='azure_v2', trace_path='/workspace/datasets/maf2/', trace_start_time='0.0.0', trace_end_time='0.0.60', trace_interval_minutes=60, trace_arrival_distribution='gamma', trace_total_rate=4.0, trace_cv=1.0, trace_map_stride=1, trace_need_sort=False, inference_architecture='sglang', migration_type='period_mig', migration_interval=10, migration_req_threshold=16, max_loras_per_batch=8, max_running_requests=16)

#Input tokens: 360334
#Output tokens: 223999

[Benchmark] 静态 LoRA 路由 (sglang):
  lora0 -> http://127.0.0.1:30001/generate
  lora1 -> http://127.0.0.1:30002/generate
  lora2 -> http://127.0.0.1:30001/generate
  lora3 -> http://127.0.0.1:30002/generate
Starting initial single prompt test run...
Test request: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.75s/it]
Initial test run completed. Starting main benchmark run...
  0%|                                                                                                                                                                                                                            | 0/1000 [00:00<?, ?it/s][189, 253, 89, 469]
last arrival: 250.5046187369351
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [04:21<00:00,  3.82it/s]

============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    8.0       
Successful requests:                     996       
Benchmark duration (s):                  261.53    
Total input tokens:                      342460    
Total generated tokens:                  218257    
Total generated tokens (retokenized):    102707    
Request throughput (req/s):              3.81      
Input token throughput (tok/s):          1309.42   
Output token throughput (tok/s):         834.52    
Total throughput (tok/s):                2143.95   
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   2042.26   
Median E2E Latency (ms):                 592.35    
---------------Time to First Token----------------
Mean TTFT (ms):                          2019.62   
Median TTFT (ms):                        592.38    
P99 TTFT (ms):                           14230.46  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          1.87      
Median TPOT (ms):                        -0.00     
P99 TPOT (ms):                           29.65     
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P99 ITL (ms):                            0.00      
==================================================


# request rate = 16

Running: python3 /workspace/sglang/benchmark/lora/lora_bench_multi.py --num-prompts 1000 --request-rate 16 --inference-architecture sglang --output-file /workspace/sglang/benchmark/lora/migration_ilp/sglang/sglang_rate16.jsonl --use-trace
Namespace(backend='sglang', base_url=None, host='0.0.0.0', port=30000, num_prompts=1000, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, request_rate=16.0, base_only=False, output_file='/workspace/sglang/benchmark/lora/migration_ilp/sglang/sglang_rate16.jsonl', seed=1, lora_assignment_strategy='sequential', worker_urls=['http://127.0.0.1:30001', 'http://127.0.0.1:30002'], use_trace=True, trace_name='azure_v2', trace_path='/workspace/datasets/maf2/', trace_start_time='0.0.0', trace_end_time='0.0.60', trace_interval_minutes=60, trace_arrival_distribution='gamma', trace_total_rate=4.0, trace_cv=1.0, trace_map_stride=1, trace_need_sort=False, inference_architecture='sglang', migration_type='period_mig', migration_interval=10, migration_req_threshold=16, max_loras_per_batch=8, max_running_requests=16)

#Input tokens: 360334
#Output tokens: 223999

[Benchmark] 静态 LoRA 路由 (sglang):
  lora0 -> http://127.0.0.1:30001/generate
  lora1 -> http://127.0.0.1:30002/generate
  lora2 -> http://127.0.0.1:30001/generate
  lora3 -> http://127.0.0.1:30002/generate
Starting initial single prompt test run...
Test request: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.75s/it]
Initial test run completed. Starting main benchmark run...
  0%|                                                                                                                                                                                                                            | 0/1000 [00:00<?, ?it/s][189, 253, 89, 469]
last arrival: 250.5046187369351
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [04:21<00:00,  3.82it/s]

============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    16.0      
Successful requests:                     996       
Benchmark duration (s):                  261.95    
Total input tokens:                      342460    
Total generated tokens:                  218257    
Total generated tokens (retokenized):    96360     
Request throughput (req/s):              3.80      
Input token throughput (tok/s):          1307.36   
Output token throughput (tok/s):         833.21    
Total throughput (tok/s):                2140.57   
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   1914.21   
Median E2E Latency (ms):                 555.07    
---------------Time to First Token----------------
Mean TTFT (ms):                          1890.32   
Median TTFT (ms):                        555.10    
P99 TTFT (ms):                           13936.52  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          1.82      
Median TPOT (ms):                        -0.00     
P99 TPOT (ms):                           28.94     
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P99 ITL (ms):                            0.00      
==================================================


# request rate = 32

Running: python3 /workspace/sglang/benchmark/lora/lora_bench_multi.py --num-prompts 1000 --request-rate 32 --inference-architecture sglang --output-file /workspace/sglang/benchmark/lora/migration_ilp/sglang/sglang_rate32.jsonl --use-trace
Namespace(backend='sglang', base_url=None, host='0.0.0.0', port=30000, num_prompts=1000, random_input_len=1024, random_output_len=128, random_range_ratio=0.0, request_rate=32.0, base_only=False, output_file='/workspace/sglang/benchmark/lora/migration_ilp/sglang/sglang_rate32.jsonl', seed=1, lora_assignment_strategy='sequential', worker_urls=['http://127.0.0.1:30001', 'http://127.0.0.1:30002'], use_trace=True, trace_name='azure_v2', trace_path='/workspace/datasets/maf2/', trace_start_time='0.0.0', trace_end_time='0.0.60', trace_interval_minutes=60, trace_arrival_distribution='gamma', trace_total_rate=4.0, trace_cv=1.0, trace_map_stride=1, trace_need_sort=False, inference_architecture='sglang', migration_type='period_mig', migration_interval=10, migration_req_threshold=16, max_loras_per_batch=8, max_running_requests=16)

#Input tokens: 360334
#Output tokens: 223999

[Benchmark] 静态 LoRA 路由 (sglang):
  lora0 -> http://127.0.0.1:30001/generate
  lora1 -> http://127.0.0.1:30002/generate
  lora2 -> http://127.0.0.1:30001/generate
  lora3 -> http://127.0.0.1:30002/generate
Starting initial single prompt test run...
Test request: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:04<00:00,  4.74s/it]
Initial test run completed. Starting main benchmark run...
  0%|                                                                                                                                                                                                                            | 0/1000 [00:00<?, ?it/s][189, 253, 89, 469]
last arrival: 250.5046187369351
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [04:20<00:00,  3.84it/s]

============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    32.0      
Successful requests:                     996       
Benchmark duration (s):                  260.32    
Total input tokens:                      342460    
Total generated tokens:                  218257    
Total generated tokens (retokenized):    95012     
Request throughput (req/s):              3.83      
Input token throughput (tok/s):          1315.52   
Output token throughput (tok/s):         838.41    
Total throughput (tok/s):                2153.93   
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   1879.18   
Median E2E Latency (ms):                 533.08    
---------------Time to First Token----------------
Mean TTFT (ms):                          1856.20   
Median TTFT (ms):                        533.11    
P99 TTFT (ms):                           13027.35  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          2.15      
Median TPOT (ms):                        -0.00     
P99 TPOT (ms):                           46.30     
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P99 ITL (ms):                            0.00      
==================================================


# request rate = 64

============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    64.0      
Successful requests:                     996       
Benchmark duration (s):                  261.91    
Total input tokens:                      342460    
Total generated tokens:                  218257    
Total generated tokens (retokenized):    98987     
Request throughput (req/s):              3.80      
Input token throughput (tok/s):          1307.56   
Output token throughput (tok/s):         833.34    
Total throughput (tok/s):                2140.90   
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   1972.36   
Median E2E Latency (ms):                 504.32    
---------------Time to First Token----------------
Mean TTFT (ms):                          1946.78   
Median TTFT (ms):                        504.39    
P99 TTFT (ms):                           13731.70  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          1.72      
Median TPOT (ms):                        -0.00     
P99 TPOT (ms):                           25.08     
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.00      
Median ITL (ms):                         0.00      
P99 ITL (ms):                            0.00      
==================================================