# v12

v11 with self-training on our half: our motif model is re-trained with pseudo-labels from v11's own ranking (top 600 suspicious, bottom 40% normal, half weight, 2-fold cross-fitted on the test week), averaged with the untrained version, then blended 50/50 with Shawn's model + motifs.
