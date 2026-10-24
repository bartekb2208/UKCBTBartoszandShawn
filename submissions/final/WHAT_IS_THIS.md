# final

A 50/50 rank blend of two models adapted to the test week. Model A (ours/) is LightGBM + ExtraTrees on 205 behavioural features plus 26 relay-motif features. Model B (shawn/) is Shawn's scenario-reconstruction model with the same motif features added by a wrapper. Both are self-trained on the test week in two rounds (round 2 co-trained on a shared teacher ranking). This is the version that scored 72.1 on the leaderboard (internal name v16). See REPORT.md for the method and evaluation.
