# eq14 — closure census (RQ10 stage A, docs/09)

Spread = log9(max/min of nonzero |v|) per accumulator row; budget span = s_x + row_span(W) + n_act + ceil(log9 fan_in).

## eq5_A_k9emb99 — k9 body / k99 embed (RTN) (norms exact)

| activation | spread p50/p99/max | zeros | log9 absmax |
|---|---|---|---|
| h_in0 | 1.77/1.77/1.77 | 0.0176 | 0.13 |
| hn0 | 1.94/2.27/2.36 | 0.0176 | 0.77 |
| attnout0 | 2.27/4.22/7.02 | 0.0 | 0.2 |
| o_pre0 | 2.57/4.56/7.54 | 0.0 | 0.22 |
| mid0 | 2.41/4.38/7.23 | 0.0 | 0.26 |
| ff0 | 4.04/6.3/10.67 | 0.0 | 0.84 |
| h_out0 | 2.45/4.4/6.05 | 0.0 | 0.61 |
| hn1 | 2.59/4.58/6.29 | 0.0 | 1.01 |
| attnout1 | 2.45/4.38/7.13 | 0.0 | 0.21 |
| o_pre1 | 2.54/4.44/6.8 | 0.0 | 0.12 |
| mid1 | 2.49/4.44/6.5 | 0.0 | 0.65 |
| ff1 | 4.04/6.21/7.78 | 0.0 | 0.69 |
| h_out1 | 2.53/4.5/6.76 | 0.0 | 0.69 |
| hn2 | 2.6/4.56/6.94 | 0.0 | 0.98 |
| attnout2 | 2.51/4.49/7.91 | 0.0 | 0.31 |
| o_pre2 | 2.48/4.43/6.45 | 0.0 | 0.34 |
| mid2 | 2.54/4.45/5.75 | 0.0 | 0.71 |
| ff2 | 4.02/6.16/9.14 | 0.0 | 0.92 |
| h_out2 | 2.55/4.43/6.07 | 0.0 | 0.78 |
| hn3 | 2.6/4.51/6.12 | 0.0 | 1.03 |
| attnout3 | 2.45/4.34/6.99 | 0.0 | 0.41 |
| o_pre3 | 2.64/4.69/6.77 | 0.0 | 0.46 |
| mid3 | 2.59/4.5/5.66 | 0.0 | 0.88 |
| ff3 | 4.01/6.19/8.15 | 0.0 | 0.95 |
| h_out3 | 2.59/4.49/6.94 | 0.0 | 1.01 |
| hn4 | 2.56/4.47/6.95 | 0.0 | 0.88 |
| attnout4 | 2.43/4.33/6.86 | 0.0 | 0.41 |
| o_pre4 | 2.45/4.32/7.08 | 0.0 | 0.51 |
| mid4 | 2.59/4.59/7.27 | 0.0 | 1.1 |
| ff4 | 4.08/6.29/9.23 | 0.0 | 1.0 |
| h_out4 | 2.61/4.59/6.35 | 0.0 | 1.24 |
| h_final | 2.61/4.59/6.35 | 0.0 | 1.24 |
| hnF | 2.75/4.71/6.44 | 0.0 | 1.38 |

| weight | row spread p50/p99/max | scale spread | k |
|---|---|---|---|
| tok_embeddings.weight | 1.77/1.77/1.77 | 0.61 | 99 |
| layers.0.attention.wq.weight | 0.63/0.63/0.63 | 1.18 | 9 |
| layers.0.attention.wk.weight | 0.63/0.63/0.63 | 0.86 | 9 |
| layers.0.attention.wv.weight | 0.63/0.63/0.63 | 0.51 | 9 |
| layers.0.attention.wo.weight | 0.63/0.63/0.63 | 0.86 | 9 |
| layers.0.feed_forward.w1.weight | 0.63/0.63/0.63 | 0.72 | 9 |
| layers.0.feed_forward.w2.weight | 0.74/0.89/0.9 | 0.64 | 9 |
| layers.0.feed_forward.w3.weight | 0.63/0.63/0.63 | 0.52 | 9 |
| layers.1.attention.wq.weight | 0.63/0.63/0.63 | 1.05 | 9 |
| layers.1.attention.wk.weight | 0.63/0.63/0.63 | 0.92 | 9 |
| layers.1.attention.wv.weight | 0.63/0.63/0.63 | 0.38 | 9 |
| layers.1.attention.wo.weight | 0.63/0.63/0.63 | 0.71 | 9 |
| layers.1.feed_forward.w1.weight | 0.63/0.63/0.63 | 0.69 | 9 |
| layers.1.feed_forward.w2.weight | 0.78/0.99/1.06 | 0.59 | 9 |
| layers.1.feed_forward.w3.weight | 0.63/0.63/0.63 | 0.45 | 9 |
| layers.2.attention.wq.weight | 0.63/0.63/0.63 | 1.26 | 9 |
| layers.2.attention.wk.weight | 0.63/0.63/0.63 | 0.71 | 9 |
| layers.2.attention.wv.weight | 0.63/0.63/0.63 | 0.47 | 9 |
| layers.2.attention.wo.weight | 0.63/0.63/0.63 | 0.61 | 9 |
| layers.2.feed_forward.w1.weight | 0.63/0.63/0.63 | 0.66 | 9 |
| layers.2.feed_forward.w2.weight | 0.75/1.01/1.04 | 0.61 | 9 |
| layers.2.feed_forward.w3.weight | 0.63/0.63/0.63 | 0.49 | 9 |
| layers.3.attention.wq.weight | 0.63/0.63/0.63 | 1.39 | 9 |
| layers.3.attention.wk.weight | 0.63/0.63/0.63 | 0.98 | 9 |
| layers.3.attention.wv.weight | 0.63/0.63/0.63 | 0.44 | 9 |
| layers.3.attention.wo.weight | 0.63/0.63/0.63 | 0.52 | 9 |
| layers.3.feed_forward.w1.weight | 0.63/0.63/0.63 | 0.66 | 9 |
| layers.3.feed_forward.w2.weight | 0.76/0.91/0.91 | 0.61 | 9 |
| layers.3.feed_forward.w3.weight | 0.63/0.63/0.63 | 0.51 | 9 |
| layers.4.attention.wq.weight | 0.63/0.63/0.63 | 1.11 | 9 |
| layers.4.attention.wk.weight | 0.63/0.63/0.63 | 0.99 | 9 |
| layers.4.attention.wv.weight | 0.63/0.63/0.63 | 0.44 | 9 |
| layers.4.attention.wo.weight | 0.63/0.63/0.63 | 0.6 | 9 |
| layers.4.feed_forward.w1.weight | 0.63/0.63/0.63 | 0.64 | 9 |
| layers.4.feed_forward.w2.weight | 0.75/0.96/1.02 | 0.67 | 9 |
| layers.4.feed_forward.w3.weight | 0.63/0.63/0.63 | 0.45 | 9 |

ops: `{"rmsnorm_eps_rel_L0": 0.000193, "rmsnorm_eps_rel_L1": 0.000108, "rmsnorm_eps_rel_L2": 6.2e-05, "rmsnorm_eps_rel_L3": 3.6e-05, "rmsnorm_eps_rel_L4": 1.9e-05, "silu_spread_map_lastL": {"w1out_spread_max": 7.06, "silu_out_spread_max": 7.36}, "softmax_mass_lastL": {"max_w": 0.286, "underflow_rows_frac": 0.0, "min_attended_p50": 1.4e-09, "mass_dynamic_range_log10": 8.3}}`

spans: `{"n_act=1": 11.25, "n_act=2": 12.25, "n_act=3": 13.25}`

## eq5_C_k63emb99 — k63 body / k99 embed (RTN) (norms exact)

| activation | spread p50/p99/max | zeros | log9 absmax |
|---|---|---|---|
| h_in0 | 1.77/1.77/1.77 | 0.0176 | 0.13 |
| hn0 | 1.94/2.27/2.36 | 0.0176 | 0.77 |
| attnout0 | 2.5/4.37/7.25 | 0.0 | 0.18 |
| o_pre0 | 2.57/4.45/6.82 | 0.0 | 0.17 |
| mid0 | 2.4/4.33/6.61 | 0.0 | 0.2 |
| ff0 | 4.07/6.27/8.25 | 0.0 | 0.88 |
| h_out0 | 2.46/4.39/6.9 | 0.0 | 0.61 |
| hn1 | 2.6/4.54/7.0 | 0.0 | 0.98 |
| attnout1 | 2.48/4.32/6.71 | 0.0 | 0.24 |
| o_pre1 | 2.53/4.4/6.48 | 0.0 | 0.09 |
| mid1 | 2.49/4.43/7.2 | 0.0 | 0.64 |
| ff1 | 4.02/6.24/9.34 | 0.0 | 0.68 |
| h_out1 | 2.53/4.54/6.67 | 0.0 | 0.66 |
| hn2 | 2.6/4.62/6.75 | 0.0 | 1.0 |
| attnout2 | 2.59/4.55/6.16 | 0.0 | 0.3 |
| o_pre2 | 2.53/4.41/6.75 | 0.0 | 0.36 |
| mid2 | 2.56/4.53/6.03 | 0.0 | 0.72 |
| ff2 | 4.01/6.07/10.64 | 0.0 | 0.98 |
| h_out2 | 2.56/4.51/7.05 | 0.0 | 0.75 |
| hn3 | 2.62/4.57/7.14 | 0.0 | 1.0 |
| attnout3 | 2.5/4.35/6.35 | 0.0 | 0.38 |
| o_pre3 | 2.5/4.51/6.75 | 0.0 | 0.34 |
| mid3 | 2.58/4.42/7.39 | 0.0 | 0.87 |
| ff3 | 4.04/6.26/9.4 | 0.0 | 1.07 |
| h_out3 | 2.59/4.5/6.23 | 0.0 | 0.98 |
| hn4 | 2.58/4.47/6.23 | 0.0 | 0.86 |
| attnout4 | 2.52/4.42/6.05 | 0.0 | 0.42 |
| o_pre4 | 2.49/4.43/6.05 | 0.0 | 0.52 |
| mid4 | 2.58/4.54/6.84 | 0.0 | 1.1 |
| ff4 | 4.1/6.27/9.26 | 0.0 | 1.06 |
| h_out4 | 2.6/4.49/6.07 | 0.0 | 1.23 |
| h_final | 2.6/4.49/6.07 | 0.0 | 1.23 |
| hnF | 2.76/4.65/6.35 | 0.0 | 1.34 |

| weight | row spread p50/p99/max | scale spread | k |
|---|---|---|---|
| tok_embeddings.weight | 1.77/1.77/1.77 | 0.61 | 99 |
| layers.0.attention.wq.weight | 1.56/1.56/1.56 | 1.18 | 63 |
| layers.0.attention.wk.weight | 1.56/1.56/1.56 | 0.86 | 63 |
| layers.0.attention.wv.weight | 1.56/1.56/1.56 | 0.51 | 63 |
| layers.0.attention.wo.weight | 1.56/1.56/1.56 | 0.86 | 63 |
| layers.0.feed_forward.w1.weight | 1.56/1.56/1.56 | 0.72 | 63 |
| layers.0.feed_forward.w2.weight | 1.67/1.82/1.84 | 0.64 | 63 |
| layers.0.feed_forward.w3.weight | 1.56/1.56/1.56 | 0.52 | 63 |
| layers.1.attention.wq.weight | 1.56/1.56/1.56 | 1.05 | 63 |
| layers.1.attention.wk.weight | 1.56/1.56/1.56 | 0.92 | 63 |
| layers.1.attention.wv.weight | 1.56/1.56/1.56 | 0.38 | 63 |
| layers.1.attention.wo.weight | 1.56/1.56/1.56 | 0.71 | 63 |
| layers.1.feed_forward.w1.weight | 1.56/1.56/1.56 | 0.69 | 63 |
| layers.1.feed_forward.w2.weight | 1.71/1.92/1.99 | 0.59 | 63 |
| layers.1.feed_forward.w3.weight | 1.56/1.56/1.56 | 0.45 | 63 |
| layers.2.attention.wq.weight | 1.56/1.56/1.56 | 1.26 | 63 |
| layers.2.attention.wk.weight | 1.56/1.56/1.56 | 0.71 | 63 |
| layers.2.attention.wv.weight | 1.56/1.56/1.56 | 0.47 | 63 |
| layers.2.attention.wo.weight | 1.56/1.56/1.56 | 0.61 | 63 |
| layers.2.feed_forward.w1.weight | 1.56/1.56/1.56 | 0.66 | 63 |
| layers.2.feed_forward.w2.weight | 1.68/1.94/1.97 | 0.61 | 63 |
| layers.2.feed_forward.w3.weight | 1.56/1.56/1.56 | 0.49 | 63 |
| layers.3.attention.wq.weight | 1.56/1.56/1.56 | 1.39 | 63 |
| layers.3.attention.wk.weight | 1.56/1.56/1.56 | 0.98 | 63 |
| layers.3.attention.wv.weight | 1.56/1.56/1.56 | 0.44 | 63 |
| layers.3.attention.wo.weight | 1.56/1.56/1.56 | 0.52 | 63 |
| layers.3.feed_forward.w1.weight | 1.56/1.56/1.56 | 0.66 | 63 |
| layers.3.feed_forward.w2.weight | 1.69/1.84/1.84 | 0.61 | 63 |
| layers.3.feed_forward.w3.weight | 1.56/1.56/1.56 | 0.51 | 63 |
| layers.4.attention.wq.weight | 1.56/1.56/1.56 | 1.11 | 63 |
| layers.4.attention.wk.weight | 1.56/1.56/1.56 | 0.99 | 63 |
| layers.4.attention.wv.weight | 1.56/1.56/1.56 | 0.44 | 63 |
| layers.4.attention.wo.weight | 1.56/1.56/1.56 | 0.6 | 63 |
| layers.4.feed_forward.w1.weight | 1.56/1.56/1.56 | 0.64 | 63 |
| layers.4.feed_forward.w2.weight | 1.68/1.9/1.95 | 0.67 | 63 |
| layers.4.feed_forward.w3.weight | 1.56/1.56/1.56 | 0.45 | 63 |

ops: `{"rmsnorm_eps_rel_L0": 0.000227, "rmsnorm_eps_rel_L1": 0.000102, "rmsnorm_eps_rel_L2": 6.3e-05, "rmsnorm_eps_rel_L3": 3.3e-05, "rmsnorm_eps_rel_L4": 1.9e-05, "silu_spread_map_lastL": {"w1out_spread_max": 6.61, "silu_out_spread_max": 6.9}, "softmax_mass_lastL": {"max_w": 0.318, "underflow_rows_frac": 0.0, "min_attended_p50": 7e-10, "mass_dynamic_range_log10": 8.66}}`

spans: `{"n_act=1": 12.17, "n_act=2": 13.17, "n_act=3": 14.17}`

## eq9_QAT_k9emb99 — k9 body / k99 embed (QAT-trained digits) (norms source-fp16 proxy)

| activation | spread p50/p99/max | zeros | log9 absmax |
|---|---|---|---|
| h_in0 | 1.77/1.77/1.77 | 0.0171 | 0.12 |
| hn0 | 1.92/2.21/2.25 | 0.0171 | 0.76 |
| attnout0 | 2.68/4.71/6.48 | 0.0 | 0.28 |
| o_pre0 | 2.55/4.56/7.0 | 0.0 | 0.15 |
| mid0 | 2.42/4.36/6.0 | 0.0 | 0.23 |
| ff0 | 4.38/6.53/9.0 | 0.0 | 0.93 |
| h_out0 | 2.48/4.46/6.15 | 0.0 | 0.73 |
| hn1 | 2.56/4.49/6.49 | 0.0 | 0.95 |
| attnout1 | 2.5/4.42/6.62 | 0.0 | 0.45 |
| o_pre1 | 2.48/4.43/8.1 | 0.0 | 0.25 |
| mid1 | 2.51/4.43/7.14 | 0.0 | 0.77 |
| ff1 | 4.06/6.19/9.64 | 0.0 | 0.91 |
| h_out1 | 2.53/4.52/7.36 | 0.0 | 0.84 |
| hn2 | 2.58/4.55/7.44 | 0.0 | 0.91 |
| attnout2 | 2.6/4.46/6.87 | 0.0 | 0.41 |
| o_pre2 | 2.46/4.45/7.43 | 0.0 | 0.44 |
| mid2 | 2.53/4.39/6.1 | 0.0 | 0.93 |
| ff2 | 4.09/6.28/9.37 | 0.0 | 1.1 |
| h_out2 | 2.55/4.5/7.35 | 0.0 | 0.98 |
| hn3 | 2.59/4.54/7.45 | 0.0 | 0.95 |
| attnout3 | 2.51/4.46/6.43 | 0.0 | 0.56 |
| o_pre3 | 2.44/4.34/7.31 | 0.0 | 0.54 |
| mid3 | 2.53/4.53/6.72 | 0.0 | 1.06 |
| ff3 | 4.1/6.21/9.25 | 0.0 | 1.11 |
| h_out3 | 2.55/4.5/8.29 | 0.0 | 1.13 |
| hn4 | 2.53/4.48/8.18 | 0.0 | 0.87 |
| attnout4 | 2.53/4.38/6.67 | 0.0 | 0.61 |
| o_pre4 | 2.49/4.53/6.92 | 0.0 | 0.79 |
| mid4 | 2.53/4.43/6.3 | 0.0 | 1.25 |
| ff4 | 4.14/6.24/8.71 | 0.0 | 1.25 |
| h_out4 | 2.6/4.58/6.05 | 0.0 | 1.45 |
| h_final | 2.6/4.58/6.05 | 0.0 | 1.45 |
| hnF | 2.73/4.77/6.26 | 0.0 | 1.32 |

| weight | row spread p50/p99/max | scale spread | k |
|---|---|---|---|
| tok_embeddings.weight | 1.77/1.77/1.77 | 0.62 | 99 |
| layers.0.attention.wq.weight | 0.63/0.63/0.63 | 1.15 | 9 |
| layers.0.attention.wk.weight | 0.63/0.63/0.63 | 0.95 | 9 |
| layers.0.attention.wv.weight | 0.63/0.63/0.63 | 0.6 | 9 |
| layers.0.attention.wo.weight | 0.63/0.63/0.63 | 0.87 | 9 |
| layers.0.feed_forward.w1.weight | 0.63/0.63/0.63 | 0.68 | 9 |
| layers.0.feed_forward.w2.weight | 0.73/0.9/0.9 | 0.66 | 9 |
| layers.0.feed_forward.w3.weight | 0.63/0.63/0.63 | 0.48 | 9 |
| layers.1.attention.wq.weight | 0.63/0.63/0.63 | 0.98 | 9 |
| layers.1.attention.wk.weight | 0.63/0.63/0.63 | 0.91 | 9 |
| layers.1.attention.wv.weight | 0.63/0.63/0.63 | 0.43 | 9 |
| layers.1.attention.wo.weight | 0.63/0.63/0.63 | 0.64 | 9 |
| layers.1.feed_forward.w1.weight | 0.63/0.63/0.63 | 0.69 | 9 |
| layers.1.feed_forward.w2.weight | 0.76/0.97/1.02 | 0.57 | 9 |
| layers.1.feed_forward.w3.weight | 0.63/0.63/0.63 | 0.45 | 9 |
| layers.2.attention.wq.weight | 0.63/0.63/0.63 | 1.15 | 9 |
| layers.2.attention.wk.weight | 0.63/0.63/0.63 | 0.7 | 9 |
| layers.2.attention.wv.weight | 0.63/0.63/0.63 | 0.42 | 9 |
| layers.2.attention.wo.weight | 0.63/0.63/0.63 | 0.61 | 9 |
| layers.2.feed_forward.w1.weight | 0.63/0.63/0.63 | 0.61 | 9 |
| layers.2.feed_forward.w2.weight | 0.76/0.99/1.03 | 0.58 | 9 |
| layers.2.feed_forward.w3.weight | 0.63/0.63/0.63 | 0.44 | 9 |
| layers.3.attention.wq.weight | 0.63/0.63/0.63 | 1.32 | 9 |
| layers.3.attention.wk.weight | 0.63/0.63/0.63 | 0.99 | 9 |
| layers.3.attention.wv.weight | 0.63/0.63/0.63 | 0.44 | 9 |
| layers.3.attention.wo.weight | 0.63/0.63/0.63 | 0.52 | 9 |
| layers.3.feed_forward.w1.weight | 0.63/0.63/0.63 | 0.59 | 9 |
| layers.3.feed_forward.w2.weight | 0.76/0.92/0.92 | 0.64 | 9 |
| layers.3.feed_forward.w3.weight | 0.63/0.63/0.63 | 0.5 | 9 |
| layers.4.attention.wq.weight | 0.63/0.63/0.63 | 1.04 | 9 |
| layers.4.attention.wk.weight | 0.63/0.63/0.63 | 1.02 | 9 |
| layers.4.attention.wv.weight | 0.63/0.63/0.63 | 0.42 | 9 |
| layers.4.attention.wo.weight | 0.63/0.63/0.63 | 0.58 | 9 |
| layers.4.feed_forward.w1.weight | 0.63/0.63/0.63 | 0.62 | 9 |
| layers.4.feed_forward.w2.weight | 0.76/0.98/0.98 | 0.67 | 9 |
| layers.4.feed_forward.w3.weight | 0.63/0.63/0.63 | 0.48 | 9 |

ops: `{"rmsnorm_eps_rel_L0": 0.000206, "rmsnorm_eps_rel_L1": 4.6e-05, "rmsnorm_eps_rel_L2": 2.4e-05, "rmsnorm_eps_rel_L3": 1.2e-05, "rmsnorm_eps_rel_L4": 6e-06, "silu_spread_map_lastL": {"w1out_spread_max": 7.38, "silu_out_spread_max": 7.68}, "softmax_mass_lastL": {"max_w": 0.398, "underflow_rows_frac": 0.0, "min_attended_p50": 5e-10, "mass_dynamic_range_log10": 8.86}}`

spans: `{"n_act=1": 11.43, "n_act=2": 12.43, "n_act=3": 13.43}`

## eq9_QAT_k27emb99 — k27 body / k99 embed (QAT-trained digits) (norms source-fp16 proxy)

| activation | spread p50/p99/max | zeros | log9 absmax |
|---|---|---|---|
| h_in0 | 1.77/1.77/1.77 | 0.0191 | 0.12 |
| hn0 | 1.93/2.23/2.33 | 0.0191 | 0.77 |
| attnout0 | 2.66/4.83/7.29 | 0.0 | 0.24 |
| o_pre0 | 2.58/4.57/7.23 | 0.0 | 0.12 |
| mid0 | 2.41/4.29/6.53 | 0.0 | 0.23 |
| ff0 | 4.24/6.41/8.9 | 0.0 | 0.84 |
| h_out0 | 2.49/4.48/6.67 | 0.0 | 0.64 |
| hn1 | 2.62/4.59/6.73 | 0.0 | 0.98 |
| attnout1 | 2.5/4.49/7.0 | 0.0 | 0.29 |
| o_pre1 | 2.5/4.5/6.7 | 0.0 | 0.15 |
| mid1 | 2.51/4.49/6.68 | 0.0 | 0.68 |
| ff1 | 4.04/6.19/8.85 | 0.0 | 0.81 |
| h_out1 | 2.54/4.57/6.36 | 0.0 | 0.71 |
| hn2 | 2.61/4.67/6.44 | 0.0 | 0.98 |
| attnout2 | 2.6/4.54/6.41 | 0.0 | 0.41 |
| o_pre2 | 2.46/4.33/6.28 | 0.0 | 0.33 |
| mid2 | 2.53/4.42/7.17 | 0.0 | 0.79 |
| ff2 | 4.08/6.2/8.43 | 0.0 | 1.07 |
| h_out2 | 2.56/4.47/6.61 | 0.0 | 0.82 |
| hn3 | 2.62/4.51/6.7 | 0.0 | 1.01 |
| attnout3 | 2.51/4.41/6.36 | 0.0 | 0.42 |
| o_pre3 | 2.44/4.42/6.24 | 0.0 | 0.38 |
| mid3 | 2.55/4.52/6.98 | 0.0 | 0.9 |
| ff3 | 4.08/6.21/8.7 | 0.0 | 1.15 |
| h_out3 | 2.54/4.5/7.01 | 0.0 | 0.99 |
| hn4 | 2.54/4.48/7.07 | 0.0 | 0.88 |
| attnout4 | 2.54/4.51/7.87 | 0.0 | 0.57 |
| o_pre4 | 2.47/4.41/6.51 | 0.0 | 0.62 |
| mid4 | 2.52/4.43/6.09 | 0.0 | 1.13 |
| ff4 | 4.11/6.21/8.83 | 0.0 | 1.14 |
| h_out4 | 2.54/4.46/6.08 | 0.0 | 1.29 |
| h_final | 2.54/4.46/6.08 | 0.0 | 1.29 |
| hnF | 2.73/4.68/6.36 | 0.0 | 1.36 |

| weight | row spread p50/p99/max | scale spread | k |
|---|---|---|---|
| tok_embeddings.weight | 1.77/1.77/1.77 | 0.59 | 99 |
| layers.0.attention.wq.weight | 1.17/1.17/1.17 | 1.09 | 27 |
| layers.0.attention.wk.weight | 1.17/1.17/1.17 | 0.93 | 27 |
| layers.0.attention.wv.weight | 1.17/1.17/1.17 | 0.53 | 27 |
| layers.0.attention.wo.weight | 1.17/1.17/1.17 | 0.87 | 27 |
| layers.0.feed_forward.w1.weight | 1.17/1.17/1.17 | 0.66 | 27 |
| layers.0.feed_forward.w2.weight | 1.26/1.42/1.44 | 0.64 | 27 |
| layers.0.feed_forward.w3.weight | 1.17/1.17/1.17 | 0.46 | 27 |
| layers.1.attention.wq.weight | 1.17/1.17/1.17 | 1.04 | 27 |
| layers.1.attention.wk.weight | 1.17/1.17/1.17 | 0.9 | 27 |
| layers.1.attention.wv.weight | 1.17/1.17/1.17 | 0.42 | 27 |
| layers.1.attention.wo.weight | 1.17/1.17/1.17 | 0.63 | 27 |
| layers.1.feed_forward.w1.weight | 1.17/1.17/1.17 | 0.68 | 27 |
| layers.1.feed_forward.w2.weight | 1.31/1.5/1.58 | 0.56 | 27 |
| layers.1.feed_forward.w3.weight | 1.17/1.17/1.17 | 0.45 | 27 |
| layers.2.attention.wq.weight | 1.17/1.17/1.17 | 1.2 | 27 |
| layers.2.attention.wk.weight | 1.17/1.17/1.17 | 0.7 | 27 |
| layers.2.attention.wv.weight | 1.17/1.17/1.17 | 0.42 | 27 |
| layers.2.attention.wo.weight | 1.17/1.17/1.17 | 0.57 | 27 |
| layers.2.feed_forward.w1.weight | 1.17/1.17/1.17 | 0.67 | 27 |
| layers.2.feed_forward.w2.weight | 1.29/1.53/1.57 | 0.59 | 27 |
| layers.2.feed_forward.w3.weight | 1.17/1.17/1.17 | 0.43 | 27 |
| layers.3.attention.wq.weight | 1.17/1.17/1.17 | 1.3 | 27 |
| layers.3.attention.wk.weight | 1.17/1.17/1.17 | 0.98 | 27 |
| layers.3.attention.wv.weight | 1.17/1.17/1.17 | 0.44 | 27 |
| layers.3.attention.wo.weight | 1.17/1.17/1.17 | 0.54 | 27 |
| layers.3.feed_forward.w1.weight | 1.17/1.17/1.17 | 0.64 | 27 |
| layers.3.feed_forward.w2.weight | 1.3/1.45/1.46 | 0.61 | 27 |
| layers.3.feed_forward.w3.weight | 1.17/1.17/1.17 | 0.49 | 27 |
| layers.4.attention.wq.weight | 1.17/1.17/1.17 | 1.07 | 27 |
| layers.4.attention.wk.weight | 1.17/1.17/1.17 | 1.03 | 27 |
| layers.4.attention.wv.weight | 1.17/1.17/1.17 | 0.44 | 27 |
| layers.4.attention.wo.weight | 1.17/1.17/1.17 | 0.54 | 27 |
| layers.4.feed_forward.w1.weight | 1.17/1.17/1.17 | 0.61 | 27 |
| layers.4.feed_forward.w2.weight | 1.3/1.5/1.51 | 0.67 | 27 |
| layers.4.feed_forward.w3.weight | 1.17/1.17/1.17 | 0.43 | 27 |

ops: `{"rmsnorm_eps_rel_L0": 0.000215, "rmsnorm_eps_rel_L1": 7.1e-05, "rmsnorm_eps_rel_L2": 5.1e-05, "rmsnorm_eps_rel_L3": 2e-05, "rmsnorm_eps_rel_L4": 1e-05, "silu_spread_map_lastL": {"w1out_spread_max": 8.18, "silu_out_spread_max": 8.49}, "softmax_mass_lastL": {"max_w": 0.343, "underflow_rows_frac": 0.0, "min_attended_p50": 1.6e-09, "mass_dynamic_range_log10": 8.34}}`

spans: `{"n_act=1": 11.83, "n_act=2": 12.83, "n_act=3": 13.83}`

compounding: `{"eq5_A_k9emb99": {"h_in0_p99": 1.77, "h_out_p99": [4.4, 4.5, 4.43, 4.49, 4.59], "o_pre_p99": [4.56, 4.44, 4.43, 4.69, 4.32], "ff_p99": [6.3, 6.21, 6.16, 6.19, 6.29]}, "eq5_C_k63emb99": {"h_in0_p99": 1.77, "h_out_p99": [4.39, 4.54, 4.51, 4.5, 4.49], "o_pre_p99": [4.45, 4.4, 4.41, 4.51, 4.43], "ff_p99": [6.27, 6.24, 6.07, 6.26, 6.27]}, "eq9_QAT_k9emb99": {"h_in0_p99": 1.77, "h_out_p99": [4.46, 4.52, 4.5, 4.5, 4.58], "o_pre_p99": [4.56, 4.43, 4.45, 4.34, 4.53], "ff_p99": [6.53, 6.19, 6.28, 6.21, 6.24]}, "eq9_QAT_k27emb99": {"h_in0_p99": 1.77, "h_out_p99": [4.48, 4.57, 4.47, 4.5, 4.46], "o_pre_p99": [4.57, 4.5, 4.33, 4.42, 4.41], "ff_p99": [6.41, 6.19, 6.2, 6.21, 6.21]}}`
