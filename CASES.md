# Worked cases (train/, out-of-fold scores)

Ranks are out-of-fold positions among all 40,131 training payments (1 = most suspicious). Wallets are relabelled W1, W2, … within each case. Hashes are truncated; full hashes are in train/. Produced by `src/cases.py`. These payments are flagged for review; that is not a finding of wrongdoing.

### Flagged: planted fan_in (easy), median rank 49

| +min | from → to | USD | asset | legs | rank | hash |
|---|---|---|---|---|---|---|
| 0.0 | W1 → W2 | 370.04 | EUR | 3 | 105 | `2D6AA3B1685D158C…` |
| 0.6 | W3 → W2 | 308.86 | EUR | 2 | 49 | `697CCC628F0AA2CF…` |
| 2.2 | W4 → W2 | 178.55 | EUR | 2 | 35 | `6CF6E1A96120957C…` |
| 3.7 | W5 → W2 | 146.12 | EUR | 2 | 36 | `DF1E00ECA06BEA31…` |
| 5.1 | W6 → W2 | 289.84 | EUR | 2 | 144 | `85AE4B94EF417936…` |

### Flagged: planted chain (easy), median rank 34

| +min | from → to | USD | asset | legs | rank | hash |
|---|---|---|---|---|---|---|
| 0.0 | W1 → W2 | 87.51 | GBP | 1 | 303 | `7A7410BF740F584E…` |
| 0.3 | W2 → W3 | 83.92 | GBP | 1 | 34 | `1AC489BA28AE6D73…` |
| 0.6 | W3 → W4 | 96.51 | GBP | 1 | 1 | `0F8A8BCBECCEAD43…` |
| 1.4 | W4 → W5 | 94.81 | GBP | 1 | 2 | `70BE56C02BF794D5…` |
| 2.7 | W5 → W6 | 101.58 | GBP | 1 | 249 | `7689C254CA8ED81A…` |

### Flagged: planted cycle (easy), median rank 8

| +min | from → to | USD | asset | legs | rank | hash |
|---|---|---|---|---|---|---|
| 0.0 | W1 → W2 | 116.87 | XRP | 1 | 22 | `4FBB5BC9F1919ACA…` |
| 0.8 | W2 → W3 | 110.83 | XRP | 1 | 3 | `3BB28E852D9C8073…` |
| 2.3 | W3 → W4 | 127.46 | XRP | 1 | 10 | `9D5E193421294E85…` |
| 3.8 | W4 → W5 | 117.17 | XRP | 1 | 4 | `F70FB74B35DE3F0D…` |
| 4.5 | W5 → W6 | 126.10 | XRP | 1 | 7 | `ADBDF3397628C05B…` |
| 5.8 | W6 → W1 | 112.48 | XRP | 1 | 15 | `3F44F5CE079686FF…` |

### Not flagged: fan-in look-alike (decoy), median rank 13243

| +min | from → to | USD | asset | legs | rank | hash |
|---|---|---|---|---|---|---|
| 0.0 | W1 → W2 | 37.17 | EUR | 2 | 13243 | `3889A4EE429390D8…` |
| 12.6 | W3 → W2 | 46.67 | EUR | 3 | 19359 | `528DCBD7606EA3F2…` |
| 15.0 | W4 → W2 | 33.28 | EUR | 2 | 16989 | `822A92A13B5F4191…` |
| 31.7 | W5 → W2 | 78.49 | EUR | 2 | 5556 | `C77EE4DF7A1FB4CF…` |
| 43.5 | W6 → W2 | 29.47 | EUR | 1 | 7536 | `F0915DBA446A8D19…` |

### Missed: planted random (medium), median rank 11896

| +min | from → to | USD | asset | legs | rank | hash |
|---|---|---|---|---|---|---|
| 0.0 | W1 → W2 | 55.78 | USD | 2 | 12399 | `399FE80A45B450CF…` |
| 24.7 | W1 → W3 | 22.62 | USD | 2 | 29739 | `BE34F2C2E6624D35…` |
| 45.3 | W1 → W4 | 45.27 | USD | 3 | 7458 | `44FF14CE6509DCF5…` |
| 79.1 | W2 → W5 | 22.10 | USD | 2 | 15398 | `8B8176212B43C4AF…` |
| 103.7 | W2 → W6 | 28.07 | GBP | 2 | 11896 | `B431151C0B517977…` |
| 158.7 | W1 → W7 | 71.97 | GBP | 2 | 7353 | `C44C6FB3A80726F1…` |
| 176.9 | W1 → W8 | 934.36 | EUR | 1 | 31756 | `B3A4428531E1D4F7…` |
| 209.5 | W5 → W4 | 18.70 | USD | 2 | 4858 | `9DCE5B3F40122A13…` |
| 256.9 | W4 → W6 | 43.63 | GBP | 2 | 9574 | `0A474948FB30D426…` |

Relay families: 20000 rows, 200 planted; median rank of planted relay rows 12258 of 40131
Top 1% (401 rows): precision 0.843; false positives by family: {'bipartite': 28, 'stack': 12, 'fan_out_in': 8, 'fan_in': 7, 'chain': 4, 'multi_stage': 3, 'aggregate_relay': 1}
