### YAML Parameter Description
The configuration instructions in the YAML file can be found below.
```yaml
Basic Config
  model_name: "deepseek_r1"                          # The model name. String type
  model_path: "/data/models/origin/DeepSeek-R1-W8A8" # The model path. String type
  exe_mode: "ge_graph"                               # The execution mode. Only support ["acl_graph", "ge_graph", "eager"]
  world_size: 16                                     # The world size. Int type

Model Config
  next_n: 1                       # Number of steps using multi-token prediction. Support [0, 1, 2, 3]
  enable_pa: False                # Whether use PageAttention. Support [False, True]
  pa_block_size: 128              # PA Block Size value. Support [128, 256] 
  enable_weight_nz: True          # Whether use nz-weight format for better performance. Support [False, True]
  enable_mla_prolog: True         # Whether use mla_prolog fusion operator. Support [False, True]
  with_ckpt: True                 # Whether load ckpt. Support [False, True]
  enable_multi_streams: True      # Whether enable multistream for better performance. Support [False, True]
  enable_profiler: True           # Whether enable profiling. Support [False, True]
  enable_cache_compile: False     # Whether enable cache compile for better performance. Support [False, True]
  enable_prefill_multi_cycle: False # Whether split prefill into multiple single batch inference. Support [False, True]
  perfect_eplb: False             # Whether enable, test uniform scenario of MoE experts. Support [False, True]
  enable_superkernel: False          # Whether enable superkernel. Support [False, True]
  enable_online_split_weight: True  # Whether enable auto-split weight. Support [False, True]
  moe_chunk_max_len: 65536          # Moe layer chunk max length. Int type
  micro_batch_mode: 0               # Whether enable prefill microbatch. Support [0, 1]. 0: Close prefill microbatch, 1: Open prefill microbatch

Data Config
  dataset: "default"  # Support ["default", "LongBench", "InfiniteBench"]
  input_max_len: 8192 # The input max length 
  max_new_tokens: 100 # The max new tokens
  batch_size: 128     # The global batch size

Parallel Config
  attn_tp_size: 1     # Attention TP Number
  dense_tp_size: 1    # Dense MLP TP Number
  moe_tp_size: 1      # MoE TP Number
  embed_tp_size: 16   # Embed TP Number
  lmhead_tp_size: 16  # LMHead TP Number
  enable_o_proj_alltoall: False  # Switch active only in prefill phase: SP-TP-SP hybrid partitioning achieved via all-to-all communication prior to o_proj computation in this phase. Support [False, True]
```