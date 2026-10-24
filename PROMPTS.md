# Prompt record (AI disclosure)

Required by the rules: model/provider, prompts, settings and reproducibility limits.
Add every new prompt at the bottom, with the London time, in the same format.

## Tools and settings

| Tool | Model / provider | Settings | Used for |
|---|---|---|---|
| Claude (claude.ai chat) | Claude Opus 5.5 (`claude-opus-5-5`), Anthropic | Default settings; web search, Google Drive and browser tools on | Reading the briefing decks, research, pipeline design, first version of the code |
| Claude Code | (fill in the model shown in Claude Code) , Anthropic | (fill in: default / any flags) | Building, running and iterating on the pipeline against the real data |

**Reproducibility limits:** LLM output is not deterministic, so re-running these prompts will not give identical code. The submitted predictions do not call any LLM: they come only from the code in this ZIP, run with `python run.py` (seed 42). The prompts are recorded so the judges can see how the code was produced.

**What the team developed vs pre-existing components:** pre-existing = pandas, numpy, scipy, scikit-learn, LightGBM, networkx, and the organisers' `payments.py` / `baseline.py` if used. Everything in `load.py`, `features.py`, `model.py`, `run.py` was written on 24 October 2026 with AI assistance as recorded below, then run, checked and modified by the team.

---

## Claude (claude.ai chat)

### 1. 06:52
Attachments: `Decode & Detect AML Training - final.pptx`, `XRPL AI Hack Training - UKCBT - October 2026.pdf` (organiser training decks)

> I am going to this hackathon tell me what it is about things I need to know and how to win and competition format, etc, read all the files completely understand them as we will need to brainstorm later

Output: summary of task, scoring, submission and rules; flagged that the XRPL Commons deck (pull your own data, unsupervised) conflicts with the final UKCBT deck (supplied labelled data).

### 2. 06:58
> Its a 2 person team this time, but I want you to think about the best way to catch all the cases easy medium and hard, to ensure I win, we need the most robust method possible and most accurate predictions, also tell me if there is a way to give more context to the pipeline to be able to win, as more context could lead to better results do research on best methods for catching all cases and let me know what needs to happen,

Output: web research (IBM Graph Feature Preprocessor, arXiv 2402.08593; Multi-GNN, arXiv 2306.11586). Recommended graph pattern features + gradient boosting + score propagation, grouped validation, two-person timeline.

### 3. 09:50
> ok so give me the final pipelines to make this best possible solution that catches everything we can in a reliable manner and how you would builf it

Output: this codebase (`config.py`, `load.py`, `features.py`, `model.py`, `run.py`), a fake-data generator in `tests/`, tested end to end on fake data only. Fixed a tie bug found in testing.

### 4. 10:16
> also have a look at these slides and see if it changes anything, keep track of prompts in a proms record md file, i will be amke claude code build the solution:
> https://docs.google.com/presentation/d/16m94Vfac2RvYfKb39_15HmN-2euafgOI/edit?usp=drivesdk&ouid=117527691890209355825&rtpof=true&sd=true

Output: read the 17-slide event deck; added this file, `TEAM.md`, `package.py` (builds the submission ZIP with all required parts), updated README.

### 5. 10:29
Attachment: photo of the briefing slide "Six ideas to build" (mule network finder, money trail tracker, limit-dodge detector, fake trading detector, alert ranker, case summary writer)

> also i got thi, do we pick a track how does thiss work? also for the submition what do we actually submit and score? and how does the leader board check our algorithm score

Output: explained the ideas are optional inspiration under one shared score. Added features for the gaps: just-under-threshold payments and their spread across linked wallets, funding-group (who funded whom) scores, cash-out to exchange-like hubs, conversion round trips. Propagation now keeps the best stage.

### 6. 10:33
> also how are we breaking down our data into features etc? and also what is in the Xrpl aml file? and what should i give claude code to build? and how do you need any dependencies from me to make this, apis or external softwares etc, also should I give you the training dataset to learn a bit more about the data so you can finalise the idea and then I can pass into claude code

Output: explanation of the feature levels, zip contents, Claude Code hand-off and dependencies (Python packages only, no APIs). Recommended sharing the training data so the loader and features can be fitted to the real schema.

### 7. 10:36
> this is the folder for the training data take your time analyse it, chck the data features we need to make etc, and validate our approach and see if we need to make changes to get the best result and win

Output: exploratory analysis of train/ (labels metadata, decoys, timing, relay cohorts), prototypes measured with grouped CV,
pipeline rebuilt for the real schema (load.py, features.py, model.py, run.py). Grouped CV AP 60.3, then ego-neighbourhood features (AP 60.4, easy 79, top-1% precision 82%).

### 8. 11:18
> make sue this is the best model and solution to ensure we can score the highest, dont get stuck on the current idea and try workarounds, evaluate it logically and pragmatically so that you can choose the best approach out there also supplement your knowledge with research if you need to to make the best highest scoring approach. - also I have attached the test dataset - also before you kick off just answer have you build the algorithm so far, so i need to make claude code do anything?

### 9. 11:22
> i have a team with a score of 65.5 on the leaderboard, recall of 30 and precision of 92 so our one needs to beat this by a huge margin

### 10. 11:40
> now the team is at 67.1 with a precision of 91 and recall of 29

### 11. 11:49
> now the current highest is at 68.7, with 92 precision and 30 recall, we need to aim for 71

### 12. 11:56
> also give me step by step instructions on how to submit V2, i have added the CVS but then what

Output (8-12): ran v2 on the test data; analysed what limits the score (fast legitimate decoys among easy cases; medium and hard near-indistinguishable by design; all metadata fields carry no signal); built v3: learned scenario reconstruction (scenarios.py), auxiliary planted-shape and relay models, decoy expert and mixture, rank blend, three submission variants. Grouped CV AP 61.6. Method report (report.md) and code ZIP.

### 13. 12:06
> so i uploaded the v3_submission.csv with the submission_code.zip but i was wondering what are the other 2 submissions csv? the misture_only and main_only, is it compliant to upload that one and are we compliant dont make changes regarding this prompt just reply to me, continue doing what you were doing

### 14. 12:23
> the mixtures only got 70.4, 96% precision and 31% recall, and the main only got 71.0, 94% precision and 30% recall, the winning one which is the v3 submission got 71.1, precision of 94% and recall of 30% these are the current stats from V3, i think we should aim for 74 now, as the other groups are catching up and we require a higher score, also side note is it beneficial to run K-fold validation(and tell me what this is).

### 15. 12:45
> also some steering, after talking to the professional here it says to get more of the observable hard and the difficult mediums, its best to really look into the data, see what we are missing do some domain related research for those issues and general in this field so to break through this ma be useful but ensure we avoid regressions each iteration should improve ideally we should hit 75 to 76 score, [v4A results: blend 70.2, selftrain 70.4, mixture 69.6, main 70.1] ... we are regressing so lets ensure we improve much more, also let me currently what is our status where we are doing good, where we are lacking, what is missing, how we can get that highest result

### 16. 12:53
> [v4B/ensemble results: ensemble 69.8, nolinks mixture 67.9, nolinks blend 68.6, nolinks selftrain 68.4] I think we started dipping after you changed how you measure the results maybe, perhaps that's the issue

### 17. 13:04
> ok carry on, but also note for the hard ones, having talked to a professional they said they will be many attributes working together to identify them, so maybe more data features also checking which ones work together to ususally make the hard cases, - have a look and continue also tell me what we are at and what we should be at going forward, really look into the data

Output (13-17): v4 (link-model degree inputs removed, mixture floor, self-training, no-links variant, rank ensemble) regressed on the leaderboard; diagnosed which change cost what; restored v3 exactly (v5) and kept self-training as a variant. Relay context analysis (feeder funding, payee forwarding, multi-day relay reconstruction): weak signal.

### 18. 13:09
> v5_submission_selftrain.csv got 71.2, 95% precision,  30% recall

Output: v6 self-training variants (2 rounds, 900 pseudo-positives); v7 floor variants isolated on the v3 base; relay-only model and within-cohort outlier test (both failed); report updated.

---

## Claude Code

### 1. (time)
> (paste the prompt here)
