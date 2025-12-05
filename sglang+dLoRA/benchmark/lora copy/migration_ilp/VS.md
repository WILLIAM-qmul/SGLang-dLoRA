root@61934144cffb:/# python /workspace/sglang/benchmark/lora/migration_ilp/auto_bench.py

Running: python3 /workspace/sglang/benchmark/lora/lora_bench_unified.py --num-prompts 1000 --trace-total-rate 10 --inference-architecture sglang --output-file /workspace/sglang/benchmark/lora/migration_ilp/sglang/sglang_rate10_stride100.jsonl --use-trace --trace-total-rate 10 --trace-map-stride 100
Namespace(backend='sglang', base_url=None, host='127.0.0.1', port=30001, num_prompts=1000, request_rate=inf, base_only=False, output_file='/workspace/sglang/benchmark/lora/migration_ilp/sglang/sglang_rate10_stride100.jsonl', seed=1, lora_assignment_strategy='sequential', worker_urls=['http://127.0.0.1:30001', 'http://127.0.0.1:30002'], use_trace=True, trace_name='azure_v2', trace_path='/workspace/datasets/maf2/', trace_start_time='0.0.0', trace_end_time='0.0.60', trace_interval_minutes=60, trace_arrival_distribution='gamma', trace_total_rate=10.0, trace_cv=1.0, trace_map_stride=100, trace_need_sort=False, inference_architecture='sglang', unified_server_url=None, num_instances=2, sglang_port=30001)

#Input tokens: 360334
#Output tokens: 223999

[Benchmark] 静态 LoRA 路由 (sglang):
  lora0 -> http://127.0.0.1:30001/generate
  lora1 -> http://127.0.0.1:30001/generate
  lora2 -> http://127.0.0.1:30002/generate
  lora3 -> http://127.0.0.1:30002/generate

Starting initial single prompt test run...
Test request: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.17s/it]
Initial test run completed. Starting main benchmark run...
  0%|                                                                                                                            | 0/1000 [00:00<?, ?it/s][147, 672, 45, 136]
last arrival: 99.20819951842402
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [03:54<00:00,  4.27it/s]

============ Serving Benchmark Result ============
Backend:                                 sglang    
Architecture:                            sglang    
Traffic request rate:                    inf       
Successful requests:                     996       
Benchmark duration (s):                  234.31    

----------- Instance Load Distribution -----------
instance_0           http://127.0.0.1:30001         Requests:  500 ( 50.0%)
instance_1           http://127.0.0.1:30002         Requests:  500 ( 50.0%)
Total                                               Requests: 1000 (100.0%)

               Load Balance Metrics               
Mean requests per instance:         500.0
Std dev:                            0.0
Coefficient of Variation (CV):      0.0%
Max/Min ratio:                      1.00

-------------- Performance Metrics ---------------
Total input tokens:                      342460    
Total generated tokens:                  218257    
Request throughput (req/s):              4.25      
Output token throughput (tok/s):         931.50    
-----------------Latency Metrics------------------
Mean E2E Latency (ms):                   55549.90  
Median TTFT (ms):                        55513.45  
Median ITL (ms):                         0.00      
==================================================

Running: python3 /workspace/sglang/benchmark/lora/lora_bench_unified.py --num-prompts 1000 --trace-total-rate 10 --inference-architecture dlora --output-file /workspace/sglang/benchmark/lora/migration_ilp/dlora/dlora_rate10_stride100.jsonl --use-trace --trace-total-rate 10 --trace-map-stride 100
Namespace(backend='sglang', base_url=None, host='127.0.0.1', port=8000, num_prompts=1000, request_rate=inf, base_only=False, output_file='/workspace/sglang/benchmark/lora/migration_ilp/dlora/dlora_rate10_stride100.jsonl', seed=1, lora_assignment_strategy='sequential', worker_urls=['http://127.0.0.1:30001', 'http://127.0.0.1:30002'], use_trace=True, trace_name='azure_v2', trace_path='/workspace/datasets/maf2/', trace_start_time='0.0.0', trace_end_time='0.0.60', trace_interval_minutes=60, trace_arrival_distribution='gamma', trace_total_rate=10.0, trace_cv=1.0, trace_map_stride=100, trace_need_sort=False, inference_architecture='dlora', unified_server_url=None, num_instances=2, sglang_port=30001)

#Input tokens: 360334
#Output tokens: 223999

[Benchmark] Using dLoRA Unified Server: http://127.0.0.1:8000
  - Routing Strategy: Least-Loaded (实时动态负载均衡)
  - Mode: Trace Replay

Starting initial single prompt test run...
Test request: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.15s/it]
Initial test run completed. Starting main benchmark run...
  0%|                                                                                                                            | 0/1000 [00:00<?, ?it/s][147, 672, 45, 136]
last arrival: 99.20819951842402
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [02:36<00:00,  6.40it/s]

[DEBUG] Manager Stats: {'num_instances': 2, 'instance_metrics': [{'instance_id': 0, 'active_requests': 0, 'total_completed': 605}, {'instance_id': 1, 'active_requests': 0, 'total_completed': 497}], 'instance_request_counts': [0, 0]}

============ Serving Benchmark Result ============
Backend:                                 sglang    
Architecture:                            dlora     
Traffic request rate:                    inf       
Successful requests:                     996       
Benchmark duration (s):                  156.30    

----------- Instance Load Distribution -----------
instance_0           http://127.0.0.1:30001         Requests:  605 ( 60.5%)
instance_1           http://127.0.0.1:30002         Requests:  497 ( 49.7%)
Total                                               Requests: 1102 (100.0%)

               Load Balance Metrics               
Mean requests per instance:         551.0
Std dev:                            54.0
Coefficient of Variation (CV):      9.8%
Max/Min ratio:                      1.22

-------------- Performance Metrics ---------------
Total input tokens:                      342460    
Total generated tokens:                  218257    
Request throughput (req/s):              6.37      
Output token throughput (tok/s):         1396.38   
-----------------Latency Metrics------------------
Mean E2E Latency (ms):                   23042.97  
Median TTFT (ms):                        23673.74  
Median ITL (ms):                         0.00      
==================================================
root@61934144cffb:/# 


root@61934144cffb:/# python /workspace/sglang/benchmark/lora/migration_ilp/auto_bench.py

Running: python3 /workspace/sglang/benchmark/lora/lora_bench_unified.py --num-prompts 1000 --trace-total-rate 10 --inference-architecture sglang --output-file /workspace/sglang/benchmark/lora/migration_ilp/sglang/sglang_rate10_stride100.jsonl --use-trace --trace-total-rate 10 --trace-map-stride 100
Namespace(backend='sglang', base_url=None, host='127.0.0.1', port=30001, num_prompts=1000, request_rate=inf, base_only=False, output_file='/workspace/sglang/benchmark/lora/migration_ilp/sglang/sglang_rate10_stride100.jsonl', seed=1, lora_assignment_strategy='sequential', worker_urls=['http://127.0.0.1:30001', 'http://127.0.0.1:30002'], use_trace=True, trace_name='azure_v2', trace_path='/workspace/datasets/maf2/', trace_start_time='0.0.0', trace_end_time='0.0.60', trace_interval_minutes=60, trace_arrival_distribution='gamma', trace_total_rate=10.0, trace_cv=1.0, trace_map_stride=100, trace_need_sort=False, inference_architecture='sglang', unified_server_url=None, num_instances=2, sglang_port=30001)

#Input tokens: 360334
#Output tokens: 223999

[Benchmark] 静态 LoRA 路由 (sglang):
  lora0 -> http://127.0.0.1:30001/generate
  lora1 -> http://127.0.0.1:30001/generate
  lora2 -> http://127.0.0.1:30002/generate
  lora3 -> http://127.0.0.1:30002/generate

Starting initial single prompt test run...
Test request: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:05<00:00,  5.25s/it]
Initial test run completed. Starting main benchmark run...
  0%|                                                                                                                                                                                                                            | 0/1000 [00:00<?, ?it/s][147, 672, 45, 136]
last arrival: 99.20819951842402
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [04:06<00:00,  4.05it/s]

============ Serving Benchmark Result ============
Backend:                                 sglang    
Architecture:                            sglang    
Traffic request rate:                    inf       
Successful requests:                     996       
Benchmark duration (s):                  246.92    

----------- Instance Load Distribution -----------
instance_0           http://127.0.0.1:30001         Requests:  819 ( 81.9%)
instance_1           http://127.0.0.1:30002         Requests:  181 ( 18.1%)
Total                                               Requests: 1000 (100.0%)

               Load Balance Metrics               
Mean requests per instance:         500.0
Std dev:                            319.0
Coefficient of Variation (CV):      63.8%
Max/Min ratio:                      4.52

-------------- Performance Metrics ---------------
Total input tokens:                      342460    
Total generated tokens:                  218257    
Request throughput (req/s):              4.03      
Output token throughput (tok/s):         883.93    
-----------------Latency Metrics------------------
Mean E2E Latency (ms):                   54181.48  
Median TTFT (ms):                        46702.21  
Median ITL (ms):                         0.00      
==================================================

Running: python3 /workspace/sglang/benchmark/lora/lora_bench_unified.py --num-prompts 1000 --trace-total-rate 10 --inference-architecture dlora --output-file /workspace/sglang/benchmark/lora/migration_ilp/dlora/dlora_rate10_stride100.jsonl --use-trace --trace-total-rate 10 --trace-map-stride 100
Namespace(backend='sglang', base_url=None, host='127.0.0.1', port=8000, num_prompts=1000, request_rate=inf, base_only=False, output_file='/workspace/sglang/benchmark/lora/migration_ilp/dlora/dlora_rate10_stride100.jsonl', seed=1, lora_assignment_strategy='sequential', worker_urls=['http://127.0.0.1:30001', 'http://127.0.0.1:30002'], use_trace=True, trace_name='azure_v2', trace_path='/workspace/datasets/maf2/', trace_start_time='0.0.0', trace_end_time='0.0.60', trace_interval_minutes=60, trace_arrival_distribution='gamma', trace_total_rate=10.0, trace_cv=1.0, trace_map_stride=100, trace_need_sort=False, inference_architecture='dlora', unified_server_url=None, num_instances=2, sglang_port=30001)

#Input tokens: 360334
#Output tokens: 223999

[Benchmark] Using dLoRA Unified Server: http://127.0.0.1:8000
  - Routing Strategy: Least-Loaded (实时动态负载均衡)
  - Mode: Trace Replay

Starting initial single prompt test run...
Test request: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:05<00:00,  5.14s/it]
Initial test run completed. Starting main benchmark run...
  0%|                                                                                                                                                                                                                            | 0/1000 [00:00<?, ?it/s][147, 672, 45, 136]
last arrival: 99.20819951842402
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [02:36<00:00,  6.38it/s]

[DEBUG] Manager Stats: {'num_instances': 2, 'instance_metrics': [{'instance_id': 0, 'active_requests': 0, 'total_completed': 524}, {'instance_id': 1, 'active_requests': 0, 'total_completed': 477}], 'instance_request_counts': [0, 0]}

============ Serving Benchmark Result ============
Backend:                                 sglang    
Architecture:                            dlora     
Traffic request rate:                    inf       
Successful requests:                     996       
Benchmark duration (s):                  156.83    

----------- Instance Load Distribution -----------
instance_0           http://127.0.0.1:30001         Requests:  524 ( 52.4%)
instance_1           http://127.0.0.1:30002         Requests:  477 ( 47.7%)
Total                                               Requests: 1001 (100.0%)

               Load Balance Metrics               
Mean requests per instance:         500.5
Std dev:                            23.5
Coefficient of Variation (CV):      4.7%
Max/Min ratio:                      1.10

-------------- Performance Metrics ---------------
Total input tokens:                      342460    
Total generated tokens:                  218257    
Request throughput (req/s):              6.35      
Output token throughput (tok/s):         1391.68   
-----------------Latency Metrics------------------
Mean E2E Latency (ms):                   18387.92  
Median TTFT (ms):                        17565.67  
Median ITL (ms):                         0.00      
==================================================
root@61934144cffb:/# 