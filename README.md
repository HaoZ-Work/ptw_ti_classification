# ptw_ti_classification

Classification task from PTW Lab

### Problem statement

### Data sets

### Approach

### Setup

### Update log

- 04.08.2022 Tried features from tsfresh again, the lgbm can not identify the negative sample at in on second data set.
  - Failed in selected feature from tsfresh. The selected feature(~6500) from first data set and second data set(~1200) are not exactly same.
- 03.08.2022 used lgbm model on fft data, achieve 0.88 on second data set
  - The extracted feature from tsfresh didn't show good f1-score. We might need feature selection and fine-tuning the lgbm model.
- 03.08.2022 rewrite the pipeline in train.py
  - > This will make the code more readable.
    >
- 01.08 create the repository
