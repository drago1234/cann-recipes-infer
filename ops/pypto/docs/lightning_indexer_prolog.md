# custom_pypto.npu\_lightning\_indexer\_prolog_pto

## 产品支持情况
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 系列产品</term>   | √  |

## 功能说明

-   算子功能：用于 Deepseek IndexerAttention 中，计算 Lightning Indexer 所需要的 query，key 和 weights。
Indexer Prolog 的量化策略如下：Q_b_proj 使用 W8A8 量化，其他 Linear 均不量化；query 使用 A8 量化，key(cache) 使用 C8 量化；反量化因子以 FP16 存储；weights 以 FP16 存储；

Query 的计算公式如下：

$$
\bold{q}, \bold{q}_{scale} = \text{DynamicQuant}(\text{Hadamard}(\text{RoPE}(\text{DeQuant}(\bold{q} \cdot \bold{w}_{qb}))))
$$

Q 的计算采用了动态的 Per-Token-Head 量化，其中 Hadamard 变换通过矩阵右乘 hadamard_q 实现。而 $\bold{q}, \bold{w}_{qb}$ 均是 Int8 类型。

Key(cache) 的计算公式如下：

$$
\bold{k}, \bold{k}_{scale} = \text{DynamicQuant}(\text{Hadamard}(\text{RoPE}(\text{LayerNorm}(\bold{x} \cdot \bold{w}_k))))
$$

Cache 的计算同样采用了动态的 Per-Token-Head 量化，其中 Hadamard 变换通过矩阵右乘 hadamard_k 实现。


Weights 的计算公式如下：

$$
\bold{weight} = (\bold{x} \cdot \bold{w}_{proj}) * \text{scale}
$$

Weights 的计算没有采用量化，同时需要最后转化为 FP16 数据类型，供后续的 Lightning Indexer 计算使用。

## 函数原型

```
custom_pypto.npu_lightning_indexer_prolog_pto(token_x, q_norm, q_norm_scale, wq_b, wq_b_scale, wk, weights_proj, ln_gamma_k, ln_beta_k, cos_idx_rope, sin_idx_rope,hadamard_q, hadamard_k, idx_k_cache, idx_k_scale_cache, idx_k_cache_index, layernorm_epsilon_k, layout_query="TND", layout_key="PA_BSND") -> (Tensor, Tensor, Tensor)
```

## 参数说明

>**说明：**<br>
>
-   **token\_x**（`Tensor`）：表示 hidden 状态，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`bfloat16`。

-   **q\_norm**（`Tensor`）：表示经过 rmsnorm 后量化的 query，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`int8`。

-   **q\_norm\_scale**（`Tensor`）：表示 query 的反量化因子，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`float32`。

-   **wq\_b**（`Tensor`）：表示 query 的权重，必选参数，不支持非连续的Tensor，数据格式支持NZ，数据类型支持`int8`。

-   **wq\_b\_scale**（`Tensor`）：表示 query 的权重反量化因子，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`float32`。

-   **wk**（`Tensor`）：表示 key 的权重，必选参数，不支持非连续的Tensor，数据格式支持NZ，数据类型支持`bfloat16`。

-   **weights_proj**（`Tensor`）：表示 weights 的权重，必选参数，不支持非连续的Tensor，数据格式支持NZ，数据类型支持`bfloat16`。

-   **ln_gamma_k**（`Tensor`）：表示 key 的 layernorm 缩放，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`bfloat16`。

-   **ln_beta_k**（`Tensor`）：表示 key 的 layernorm 偏移，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`bfloat16`。

-   **cos_idx_rope**（`Tensor`）：表示用于 RoPE 的 cos，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`。

-   **sin_idx_rope**（`Tensor`）：表示用于 RoPE 的 sin，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`

-   **hadamard_q**（`Tensor`）：表示用于 query Hadamard 变换的权重矩阵，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`。

-   **hadamard_k**（`Tensor`）：表示用于 key Hadamard 变换的权重矩阵，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`。

-   **idx_k_cache**（`Tensor`）：表示 key 的缓存，必选参数，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`int8`。

-   **idx_k_scale_cache**（`Tensor`）：表示 key 反量化因子的缓存，必选参数，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`float16`。

-   **idx_k_cache_index**（`Tensor`）：表示更新 key 缓存的位置，必选参数，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`int64`。

-   **layernorm_epsilon_k**（`float`）：表示 key layernorm 防除 0 系数，必选参数，数据类型支持`float32`。

-   **layout\_query**（`str`）：可选参数，用于标识输入`query`的数据排布格式，默认值"TND"。当前仅支持 "TND"。

-   **layout\_key**（`str`）：可选参数，用于标识输入`key`的数据排布格式，默认值"PA_BSND"。当前仅支持 "PA_BSND"。

## 返回值说明

-   **query**（`Tensor`）：公式中 query 的输出 tensor，数据格式支持 ND，数据类型支持`int8`。

-   **query_scale**（`Tensor`）：公式中 query 反量化因子的输出 tensor，数据格式支持 ND，数据类型支持`float16`。

-   **weights**（`Tensor`）：公式中 weights 的输出 tensor，数据格式支持 ND，数据类型支持`float16`。

## 约束说明

-   该接口支持推理场景下使用。
-   该接口支持图模式（PyTorch 2.1版本）。
-   该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。

## 算子代码执行示例
-   算子源码执行参考[test_lightning_indexer_prolog.py](../examples/test_lightning_indexer_prolog.py)