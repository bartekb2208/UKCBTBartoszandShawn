# v6

Our model A plus the observable-hard relay-motif features (aggregate relay: inbound burst, 3.5-5.5 h delay, outbound burst, conserved value, repeated; distributed relay: feeder burst, about 2 h, repeated bridge A->B, about 2 h, payout burst). LightGBM x3 + ExtraTrees. Train CV 62.99 (v2 without motifs: 62.34). The hard max(base, motif) boost from the brief is not used: it lowered CV (motif-tagged rows are only 3-4% suspicious).
