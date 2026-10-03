# EDA Summary - Final IoT Network Intrusion Dataset

## 1. Dataset Overview
- Raw dataset size: **625,783 rows x 86 columns**
- Numerical model features identified: **79**
- Categorical feature columns: **0**
- Primary binary target: **`Label`**
- Multi-class targets: **`Cat`** and **`Sub_Cat`**

## 2. Data Quality Findings
- Raw missing values: **0**
- Raw infinite numerical values: **736**
- Exact duplicate rows: **164,087**
- Duplicate percentage: **26.22%**
- Cleaned EDA rows after exact duplicate removal: **461,696**
- Remaining infinite values after cleaning: **0**

Infinite values were converted to `NaN` rather than `0`, because zero is a valid network
measurement and should not be invented during EDA.

## 3. Target Balance

### Raw `Label`
| Label | Percentage |
| --- | --- |
| Anomaly | 93.6 |
| Normal | 6.4 |

### Cleaned `Label`
| Label | Percentage |
| --- | --- |
| Anomaly | 91.64 |
| Normal | 8.36 |

The dataset is clearly **class-imbalanced**, with substantially more anomalous traffic than normal
traffic. For later model evaluation, metrics such as precision, recall, F1-score, and per-class
performance will be more informative than accuracy alone.

## 4. Attack Categories
The cleaned data contains **5** values in `Cat` and
**9** values in `Sub_Cat`.

### `Cat`
| Cat | Count | Percentage |
| --- | --- | --- |
| Mirai | 281102 | 60.88 |
| DoS | 59390 | 12.86 |
| Scan | 56744 | 12.29 |
| Normal | 38598 | 8.36 |
| MITM ARP Spoofing | 25862 | 5.6 |

### `Sub_Cat`
| Sub_Cat | Count | Percentage |
| --- | --- | --- |
| Mirai-UDP Flooding | 142217 | 30.8 |
| Mirai-Hostbruteforceg | 86507 | 18.74 |
| DoS-Synflooding | 59390 | 12.86 |
| Scan Port OS | 39811 | 8.62 |
| Normal | 38598 | 8.36 |
| Mirai-HTTP Flooding | 26539 | 5.75 |
| MITM ARP Spoofing | 25862 | 5.6 |
| Mirai-Ackflooding | 25839 | 5.6 |
| Scan Hostport | 16933 | 3.67 |

## 5. Constant Features
The following features contain only one value in the cleaned dataset and carry no useful variation
for modeling:

`Fwd_PSH_Flags`, `Fwd_URG_Flags`, `Fwd_Byts/b_Avg`, `Fwd_Pkts/b_Avg`, `Fwd_Blk_Rate_Avg`, `Bwd_Byts/b_Avg`, `Bwd_Pkts/b_Avg`, `Bwd_Blk_Rate_Avg`, `Init_Fwd_Win_Byts`, `Fwd_Seg_Size_Min`

## 6. Strongest Numerical Relationships with `Label`
`Label` was encoded only for this EDA as `Normal = 0` and `Anomaly = 1`.

- `Dst_Port`: -0.4806
- `Fwd_Pkt_Len_Std`: -0.3362
- `ACK_Flag_Cnt`: -0.2988
- `Src_Port`: 0.2560
- `Fwd_Pkt_Len_Max`: -0.2232
- `Protocol`: 0.2214
- `TotLen_Fwd_Pkts`: -0.1831
- `Subflow_Fwd_Byts`: -0.1831
- `Pkt_Len_Std`: -0.1785
- `Fwd_Seg_Size_Avg`: -0.1753

A positive value means the feature tends to increase with the encoded Anomaly class; a negative
value means larger values are associated more with Normal traffic. **Correlation measures only
linear association and does not prove causation.**

## 7. Identifier and Leakage Caution
`Flow_ID`, `Src_IP`, `Dst_IP`, and raw `Timestamp` should not be passed directly into a baseline
model. They are high-cardinality metadata that can cause memorization or leakage; if used later,
they should first be transformed into engineered features (e.g. IP subnet, time-of-day bucket).

## 8. Main EDA Conclusions
This IoT intrusion dataset is large, mostly numerical, and suitable for supervised attack-detection
experiments. Before modeling, the key preprocessing decisions are:

1. remove or deliberately handle exact duplicates;
2. replace non-finite numerical values;
3. remove constant columns;
4. decide whether the task is binary (`Label`) or multi-class (`Cat` / `Sub_Cat`);
5. handle class imbalance correctly;
6. build a preprocessing pipeline that prevents train/test leakage.

All plots, tables, the PDF report, and this summary are stored in the `outputs` folder.