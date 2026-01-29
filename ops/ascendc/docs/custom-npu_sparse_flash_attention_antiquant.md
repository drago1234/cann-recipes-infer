# custom.npu\_sparse\_flash\_attention\_antiquant<a name="ZH-CN_TOPIC_0000001979260729"></a>

## 产品支持情况 <a name="zh-cn_topic_0000001832267082_section14441124184110"></a>
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 推理系列产品</term>   | √  |

## 功能说明<a name="zh-cn_topic_0000001832267082_section14441124184110"></a>

`SparseFlashAttentionAntiquant`在[SparseFlashAttention](./custom-npu_sparse_flash_attenton.md)的基础上支持了[Per-Token-Head-Tile-128量化](../../../docs/models/deepseek-v3.2-exp/deepseek_v3.2_exp_inference_guide.md)输入。随着大模型上下文长度的增加，Sparse Attention的重要性与日俱增，这一技术通过“只计算关键部分”大幅减少计算量，然而会引入大量的离散访存，造成数据搬运时间增加，进而影响整体性能。Sparse Attention计算可表示为:
$$
\text{softmax}(\frac{Q @ \text{Dequant}({\tilde{K}^{INT8}},{Scale_K})^T}{\sqrt{d_k}})@\text{Dequant}(\tilde{V}^{INT8},{Scale_V}),
$$
其中$\tilde{K},\tilde{V}$为基于某种选择算法(如`LightningIndexer`)得到的重要性较高的Key和Value，一般具有稀疏或分块稀疏的特征，$d_k$为$Q,\tilde{K}$每一个头的维度，$\text{Dequant}(\cdot,\cdot)$为反量化函数。
本次公布的`SparseFlashAttentionAntiquant`是面向Sparse Attention的全新算子，针对离散访存进行了指令缩减及搬运聚合的细致优化。

## 函数原型<a name="zh-cn_topic_0000001832267082_section45077510411"></a>

```
custom.npu_sparse_flash_attention_antiquant(Tensor query, Tensor key, Tensor value, Tensor sparse_indices, float scale_value, int sparse_block_size, int key_quant_mode, int value_quant_mode, *, Tensor? key_dequant_scale=None, Tensor? value_dequant_scale=None, Tensor? block_table=None, Tensor? actual_seq_lengths_query=None, Tensor? actual_seq_lengths_kv=None, str layout_query='BSND', str layout_kv='BSND', int sparse_mode=3, int attention_mode=0, int quant_scale_repo_mode=0, int tile_size=0, int rope_head_dim=0) -> Tensor
```

## 参数说明<a name="zh-cn_topic_0000001832267082_section112637109429"></a>

>**说明：**<br> 
>
>- query、key、value参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
>- Q\_S和S1表示query shape中的S，KV\_S和S2表示key shape中的S，Q\_N表示num\_query\_heads，KV\_N表示num\_key\_value\_heads。

-   **query**（`Tensor`）：必选参数，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，query相同dtype的q_nope和q_rope按D维度拼接得到。 

-   **key**（`Tensor`）：必选参数，不支持非连续，数据格式支持ND，数据类型支持`int8`，`int8`的k_nope、query相同dtype的k_rope和`float32`的量化参数按D维度拼接得到，layout\_kv为PA\_BSND时shape为[block\_num, block\_size, KV\_N, D]，其中block\_num为PageAttention时block总数，block\_size为一个block的token数。

-   **value**（`Tensor`）：必选参数，不支持非连续，数据格式支持ND，数据类型支持`int8`。
    
-   **sparse\_indices**（`Tensor`）：必选参数，代表离散取kvCache的索引，不支持非连续，数据格式支持ND,数据类型支持`int32`，shape需要传入[B, Q\_S, KV\_N, sparse\_size]，其中sparse\_size为一次离散选取的token数，需要保证每行有效值均在前半部分，无效值均在后半部分。

-   **scale\_value**（`double`）：必选参数，代表缩放系数，作为query和key矩阵乘后Muls的scalar值，数据类型支持`float`。

-   **sparse\_block\_size**（`int`）：必选参数，代表sparse阶段的block大小，在计算importance score时使用，数据类型支持`int64`。  
    
-   **key\_quant\_mode**（`int`）：必选参数，代表key的量化模式，数据类型支持`int64`，支持传入2，代表per_tile量化模式。  
    
-   **value\_quant\_mode**（`int`）：必选参数，代表value的量化模式，数据类型支持`int64`，支持传入2，代表per_tile量化模式。  

- <strong>*</strong>：代表其之前的参数是位置相关的，必须按照顺序输入，属于必选参数；其之后的参数是键值对赋值，与位置无关，属于可选参数（不传入会使用默认值）。

-   **key_dequant_scale**（`Tensor`）：可选参数，预留参数，当前不支持。

-   **value_dequant_scale**（`Tensor`）：可选参数，预留参数，当前不支持。

-   **block\_table**（`Tensor`）：可选参数，表示PageAttention中kvCache存储使用的block映射表。数据格式支持ND，数据类型支持`int32`，shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的s2对应的block数量，即s2\_max / block\_size向上取整。

-   **actual\_seq\_lengths\_query**（`Tensor`）：可选参数，表示不同Batch中`query`的有效token数，数据类型支持`int32`。如果不指定seqlen可传入None，表示和`query`的shape的S长度相同。
    >该入参中每个Batch的有效token数不超过`query`中的维度S大小。支持长度为B的一维tensor。当`query`的input\_layout为TND时，该入参必须传入，且以该入参元素的数量作为B值，该入参中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须>=前一个元素的值。不能出现负值。

-   **actual\_seq\_lengths\_kv**（`Tensor`）：可选参数，表示不同Batch中`key`和`value`的有效token数，数据类型支持`int32`。如果不指定None，表示和key的shape的S长度相同。
    >该入参中每个Batch的有效token数不超过`key/value`中的维度S大小且不小于0。支持长度为B的一维tensor。

-   **layout\_query**（`str`）：可选参数，用于标识输入`query`的数据排布格式，用户不特意指定时可传入默认值"BSND"，支持传入BSND和TND。

    >**说明**：
       >1、query数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示hidden层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。

-   **layout\_kv**（`str`）：可选参数，用于标识输入`key`的数据排布格式，用户不特意指定时可传入默认值"BSND"，支持传入PA\_BSND，PA\_BSND在使能PageAttention时使用。

-   **sparse\_mode**（`int`）：可选参数，表示sparse的模式。数据类型支持`int64`。
    -   sparse\_mode为0时，代表全部计算。
    -   sparse\_mode为3时，代表rightDownCausal模式的mask，对应以右下顶点往左上为划分线的下三角场景。

-   **attention\_mode**（`int`）：可选参数，表示attention的模式。数据类型支持`int64`，支持传入2，表示MLA-absorb模式，即QK的D包含rope和nope两部分，且KV是同一份，默认值为0。

-   **quant\_scale\_repo\_mode**（`int`）：可选参数，表示量化参数的存放模式。数据类型支持`int64`，支持传入1，表示combine模式，即量化参数和数据混合存放，默认值为0。

-   **tile\_size**（`int`）：可选参数，表示per_tile时每个参数对应的数据块大小，仅在per_tile时有效。数据类型支持`int64`，默认值为0。

-   **rope\_head\_dim**（`int`）：可选参数，表示MLA架构下的rope head dim大小，仅在attention\_mode为2时有效。数据类型支持`int64`，默认值为0。

## 返回值说明<a name="zh-cn_topic_0000001832267082_section22231435517"></a>

-   **out**（`Tensor`）：公式中的输出。数据格式支持ND，数据类型支持`bfloat16`，输出shape[B,S1,N1,D];[T1,B1,D]。

## 约束说明<a name="zh-cn_topic_0000001832267082_section12345537164214"></a>

-   该接口支持推理场景下使用。
-   该接口支持图模式。
-   该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- 参数query中的N支持1/2/4/8/16/32/64/128，key、value的N支持1
- 参数query中的D值为576，即nope\+rope=512\+64。
- 参数key、value中的D值为656，即nope\+rope\*2\+dequant\_scale\*4=512\+64\*4\+4\*4。
- 支持block\_size取值为16的整数倍，最大支持到1024。
- sparse\_block\_size支持范围为1至16。
- 支持sparse\_block\_size整除block\_size。
- 支持tile\_size支持取值为128。
- 支持rope\_head\_dim支持取值为64。
- layout\_query为TND且layout\_kv为BSND场景不支持。
## 调用示例<a name="zh-cn_topic_0000001832267082_section14459801435"></a>

-   详见[test_npu_sparse_flash_attention.py](../examples/test_npu_sparse_flash_attention.py)