# IoT Intrusion Detection - Preprocessing Summary

## 1. Raw Dataset
- Rows: **625,783**
- Columns: **86**

## 2. Duplicate Handling
- Duplicate rows detected: **164,087**
- Duplicate percentage: **26.22%**
- Rows remaining: **461,696**

## 3. Prediction Target
```
Label
Normal -> 0
Anomaly -> 1
```

## 4. Leakage Prevention
Excluded from the feature matrix `X`:
- `Cat`, `Sub_Cat` - alternative targets; they describe the attack type, which is only knowable
  once a flow is already known to be anomalous, so using them as predictors would leak the answer.
- `Flow_ID`, `Src_IP`, `Dst_IP`, `Timestamp` - high-cardinality identifiers that risk memorization
  and poor generalization rather than learning general traffic behavior.

## 5. Train/Test Split
- Training rows: **369,356**
- Testing rows: **92,340**
- Split: **80% / 20%**
- `random_state=42`, `stratify=y`

## 6. Non-Finite Values
- Infinite values converted -> train: **498**, test: **148**
- Affected columns: `Flow_Byts/s`, `Flow_Pkts/s`
- Converted to `NaN` (never to `0`, which is a valid network measurement)

## 7. Missing Values
- Columns with missing values in X_train (after infinity conversion): **2**
- Total missing values in X_train: **498**
- Total missing values in X_test (before imputation): **148**

## 8. Imputation
- Numerical -> training median (`SimpleImputer(strategy="median")`, fit on X_train only)
- Categorical -> training mode, only if categorical predictors exist (none in this dataset)

## 9. Missing Indicators
Indicator columns created: `Flow_Byts/s_missing`, `Flow_Pkts/s_missing`

## 10. Constant Features
Removed constant columns (10): `Fwd_PSH_Flags`, `Fwd_URG_Flags`, `Fwd_Byts/b_Avg`, `Fwd_Pkts/b_Avg`, `Fwd_Blk_Rate_Avg`, `Bwd_Byts/b_Avg`, `Bwd_Pkts/b_Avg`, `Bwd_Blk_Rate_Avg`, `Init_Fwd_Win_Byts`, `Fwd_Seg_Size_Min`

## 11. Encoding
Target:
```
Normal = 0
Anomaly = 1
```
No categorical predictor encoding was required after identifier and target-leakage columns were excluded — every remaining predictor in this dataset is numeric.

## 12. Scaling
```
StandardScaler
Fit on training data only (69 continuous features)
Applied unchanged to test data
```

## 13. Final Dataset Shapes
- X_train_processed: **(369356, 71)**
- X_test_processed: **(92340, 71)**
- y_train_encoded: **(369356,)**
- y_test_encoded: **(92340,)**
- Final number of predictors: **71**

## 14. Final Validation
- Remaining NaN -> train: **0**, test: **0**
- Remaining infinity -> train: **0**, test: **0**
- Train/test column alignment: **True**
- Target classes: **[0, 1]**

## 15. Final Conclusion
The processed training and testing datasets contain only numeric, leakage-free predictors with no
missing or infinite values, and are ready for feature optimization and model training.