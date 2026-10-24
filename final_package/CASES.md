# Worked cases (train/, out-of-fold scores of model A)

Ranks are out-of-fold positions among all 40,131 training payments (1 = most suspicious), from `ours/src/cv.py` (model A, 5-fold cross-validation). Wallets are relabelled W1, W2, ... within each case. Hashes are shortened; the full hashes are in the organiser's `train/transactions.parquet`. Produced by `ours/src/cases.py`. The data is synthetic; these payments are "flagged for review" examples, not findings about any account.

What each case shows: the three flagged cases are easy planted shapes (fan-in, chain, cycle) that run at about one step per minute between first-time counterparties; the "not flagged" case has the same fan-in shape at a slower, irregular pace and is a legitimate look-alike; the "missed" case is a planted medium-difficulty scenario that looks like ordinary activity and ranks far down.

### Flagged: planted fan_in (easy), median rank 38

| +min | from → to | USD | asset | legs | rank | hash |
|---|---|---|---|---|---|---|
| 0.0 | W1 → W2 | 370.04 | EUR | 3 | 147 | `2D6AA3B1685D158C…` |
| 0.6 | W3 → W2 | 308.86 | EUR | 2 | 38 | `697CCC628F0AA2CF…` |
| 2.2 | W4 → W2 | 178.55 | EUR | 2 | 35 | `6CF6E1A96120957C…` |
| 3.7 | W5 → W2 | 146.12 | EUR | 2 | 21 | `DF1E00ECA06BEA31…` |
| 5.1 | W6 → W2 | 289.84 | EUR | 2 | 183 | `85AE4B94EF417936…` |

### Flagged: planted chain (easy), median rank 108

| +min | from → to | USD | asset | legs | rank | hash |
|---|---|---|---|---|---|---|
| 0.0 | W1 → W2 | 329.13 | XRP | 1 | 151 | `0F49046B199684D9…` |
| 0.3 | W2 → W3 | 347.12 | XRP | 1 | 108 | `016CC224A6DE4C86…` |
| 0.7 | W3 → W4 | 308.47 | XRP | 1 | 82 | `2D8603931FD8E1DE…` |
| 0.9 | W4 → W5 | 290.43 | XRP | 1 | 19 | `E66D3B3CEB37AAAD…` |
| 1.7 | W5 → W6 | 292.69 | XRP | 1 | 74 | `21A6952C44D8C006…` |
| 2.2 | W6 → W7 | 283.74 | XRP | 1 | 219 | `7E5B8BA015666DBC…` |
| 2.9 | W7 → W8 | 258.91 | XRP | 1 | 1185 | `16141DCA23308E5C…` |

### Flagged: planted cycle (easy), median rank 12

| +min | from → to | USD | asset | legs | rank | hash |
|---|---|---|---|---|---|---|
| 0.0 | W1 → W2 | 116.87 | XRP | 1 | 59 | `4FBB5BC9F1919ACA…` |
| 0.8 | W2 → W3 | 110.83 | XRP | 1 | 10 | `3BB28E852D9C8073…` |
| 2.3 | W3 → W4 | 127.46 | XRP | 1 | 8 | `9D5E193421294E85…` |
| 3.8 | W4 → W5 | 117.17 | XRP | 1 | 6 | `F70FB74B35DE3F0D…` |
| 4.5 | W5 → W6 | 126.10 | XRP | 1 | 13 | `ADBDF3397628C05B…` |
| 5.8 | W6 → W1 | 112.48 | XRP | 1 | 40 | `3F44F5CE079686FF…` |

### Not flagged: fan-in look-alike (decoy), median rank 12010

| +min | from → to | USD | asset | legs | rank | hash |
|---|---|---|---|---|---|---|
| 0.0 | W1 → W2 | 37.17 | EUR | 2 | 12010 | `3889A4EE429390D8…` |
| 12.6 | W3 → W2 | 46.67 | EUR | 3 | 18683 | `528DCBD7606EA3F2…` |
| 15.0 | W4 → W2 | 33.28 | EUR | 2 | 14562 | `822A92A13B5F4191…` |
| 31.7 | W5 → W2 | 78.49 | EUR | 2 | 7236 | `C77EE4DF7A1FB4CF…` |
| 43.5 | W6 → W2 | 29.47 | EUR | 1 | 8451 | `F0915DBA446A8D19…` |

### Missed: planted random (medium), median rank 9990

| +min | from → to | USD | asset | legs | rank | hash |
|---|---|---|---|---|---|---|
| 0.0 | W1 → W2 | 55.78 | USD | 2 | 15289 | `399FE80A45B450CF…` |
| 24.7 | W1 → W3 | 22.62 | USD | 2 | 28602 | `BE34F2C2E6624D35…` |
| 45.3 | W1 → W4 | 45.27 | USD | 3 | 6977 | `44FF14CE6509DCF5…` |
| 79.1 | W2 → W5 | 22.10 | USD | 2 | 15944 | `8B8176212B43C4AF…` |
| 103.7 | W2 → W6 | 28.07 | GBP | 2 | 9990 | `B431151C0B517977…` |
| 158.7 | W1 → W7 | 71.97 | GBP | 2 | 8769 | `C44C6FB3A80726F1…` |
| 176.9 | W1 → W8 | 934.36 | EUR | 1 | 33345 | `B3A4428531E1D4F7…` |
| 209.5 | W5 → W4 | 18.70 | USD | 2 | 4967 | `9DCE5B3F40122A13…` |
| 256.9 | W4 → W6 | 43.63 | GBP | 2 | 7509 | `0A474948FB30D426…` |

Relay families: 20000 rows, 200 planted; median rank of planted relay rows 12128 of 40131
Top 1% (401 rows): precision 0.830; false positives by family: {'bipartite': 26, 'stack': 12, 'fan_out_in': 8, 'chain': 8, 'fan_in': 7, 'multi_stage': 5, 'aggregate_relay': 2}
