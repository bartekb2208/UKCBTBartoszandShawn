# AI use disclosure (terms cl. 3.3, 3.4 and 6.1)

**No AI model runs when the predictions are made.** `predictions.csv` comes only from the code in this ZIP (`run_v5.sh`). That code makes no LLM, API or network call and uses no pre-trained model or weights; both models are trained from scratch on the organiser's `train/` data in every run.

AI assistants were used to plan and write the code, and to run the experiments. The team remains responsible for the AI-generated material. Two parts of the code were written with AI help in two separate tools:

| Part | Written by | Tool | Record |
|---|---|---|---|
| `ours/` (model A, relay-motif detectors, self-training, the wrappers that add motif features and an external teacher to model B), the blend scripts, `run.sh`, this ZIP's documents | Bartosz Bielecki with an AI assistant | Claude Code | section A below |
| `shawn/` (model B) | Shawn Dsouza with an AI assistant | Claude (claude.ai chat) | `shawn/PROMPTS.md` (section B) |

## A. Claude Code (model A and packaging)

- **Provider and tool:** Anthropic, Claude Code (VS Code extension), 24 October 2026. Permission mode "auto". Sampling settings were not set by us (tool defaults); the tool chose its own reasoning-effort level per turn.
- **Models:** Claude Opus 5.5 (`claude-opus-5-5`) for most of the session. Claude Sonnet 5.5 (`claude-sonnet-5-5`) for at least the turn that assembled the `submissions/` folder (about 12:45 to 12:50). The `/model` command was used to select Claude Fable 5.1 (`claude-fable-5-1`) and then straight back to Opus 5.5 at about 12:55. [FILL: Bartosz to confirm that no work was done while Fable 5.1 was selected.]
- **What the assistant did:** read the organiser README, starter notebook and `payments.py`; profiled the training data; wrote the code in `ours/`; ran the cross-validation experiments and chose features from fold scores; read the teammate's code and ran it unchanged to check it reproduces its own submitted file; wrote the blend script and the documents in this ZIP.
- **What the team did:** set the goal, supplied the plan documents, placed the data, uploaded each candidate file to the leaderboard and reported the scores, and made every decision about what to submit.
- **Data shared with the assistant:** the organiser's train, validation and test files were read by code running on the team's own laptop; the assistant saw sample rows, counts and aggregates in the session, which are sent to Anthropic as part of the conversation. The team holds no test labels. The data is synthetic and contains no personal data.
- **Prompts** (the user's messages to Claude Code, verbatim and in order; times are approximate, London time):
  1. (about 10:55, with `PLAN.md` and `PRIMER.md` attached) "Look through the entire repo, read everything especially the plan and primer, and execute. The objective variable we're trying to optimise is the score output"
  2. (about 12:17) "another folder has been uploaded, the test data, my friends model hits 71.1 on the leaderboard thats the baseline now that we need to beat."
  3. (about 12:33) "Pull from the git, my teammate has uploaded their work, analyse it deepy and tell me if you need me anything to do. Dont push or change anything about the git for now, we'll handle this later, dont do any git pushes just work locally for now. Analyse and look for paths for improvement."
  4. (about 12:48, with a screenshot of the submission page) "This is the submission screen, for what you want me to test, make a folder called submissions and add them into v1, v2, v3 etc and in those put the code and predictions csv in and ill upload to test the predictions."
  5. (about 12:52) "whats currently running in the background?"
  6. (about 13:00, with a screenshot of the organiser's rules slide) "Does v5, the most successful model scored 71.3 do an audit on the model / code to make sure it is compliant with the rules."
  7. (about 13:25, with a pasted design brief titled "Implement an 'observable hard relay' feature family", which describes the aggregate and distributed relay detectors, their windows and ratio bands, the per-transaction features and a max(base, motif) boost) "For Obersvable hard, try and push towards this pattern design, make a v5 model featuring this if it doesnt work then scrap:"
  8. (about 13:27) "well, dont scrap, but save as a submission and then we'll give it a shot"
  9. (about 13:30) "scrap the audit for now, we'll do that later, we need to focus on winning"
  10. (about 13:36) "Dont compact, sorry for interrupting, go from where u were"
  11. (about 13:40) "make sure all your background processes are still running"
  12. (about 13:48) leaderboard results for v6, v7 and v8 ("v6 scores 69.7 ... v7 scores best, 75.6, previously shawns score achieved 71.5, Ill tell him to push that soon. ... v8 scores 75.5 ..."). [FILL: confirm v7's score; later results suggest 71.6.]
  13. (about 14:00) leaderboard results for v9, v10 and v11 (71.6, 71.4, 71.7), and "Shawn has pushed, ive pulled to you should see shawns files somewhere in the folder, if not then check it"
  14. (about 14:02) "Just so you know, top scores on the leaderboard is 72.3 with precision of 94% and recall of 30% by the way"
  15. (about 14:08, with a pasted ChatGPT response recommending a two-stage detector: motif candidates, then a rejector trained on look-alike candidates with account-baseline features, validated by hub) "This is Chatgpt's response to the problem of similar patterns for observable hard fraudulent and non fraudulent following the same pattern you spotted from earlier, do with it what you will:"
  16. (about 14:23) leaderboard results for v12, v13 and v14 (71.8, 72.0, 72.0)
  17. (about 14:40) leaderboard results for v15, v16 and v17 (72.0, 72.1, 71.1), the top score (72.6), and "We have until 4:30 we do not stop we chug on."
- **How leaderboard feedback was used:** after each batch of uploads the team reported the scores, and the next variants were chosen from them. Every variant is produced by the code in this ZIP.
- **Session commands:** the team ran `/model` (see Models above) and one `/compact`, which was cancelled.
- **Reproducibility limits:** LLM output is not deterministic, so repeating these prompts will not give identical code. The predictions do not depend on any LLM: they are reproduced by running `run_v5.sh` (fixed seeds, pinned libraries; repeated runs gave byte-identical files).

## B. Claude chat (Shawn, model B)

See `shawn/PROMPTS.md`: provider, model, settings, every prompt and the reproducibility limits for the planning, research and first versions of the code. [FILL: Shawn to complete the "Claude Code" row and section of `shawn/PROMPTS.md`, which still contain template placeholders, or delete them if that tool was not used.]

## C. Other AI tools

- **ChatGPT (OpenAI):** the team consulted ChatGPT for ideas and pasted its answers into the Claude Code session (prompts 7 and 15 above): the relay-motif design brief [FILL: confirm that the brief in prompt 7 also came from ChatGPT] and the two-stage-detector note. The motif detectors in `ours/src/dd/motifs.py` follow that brief; the two-stage rejector was tested and not used. [FILL: ChatGPT model and settings, and the prompts the team gave it.]
- [FILL: any other AI tool (for example OpenAI Codex, which the repository's `AGENTS.md` refers to): model, provider, settings, prompts and the files it changed; or "None".]

## Planning documents

The plan and primer files that started the build session were drafted in an earlier AI planning chat on the same day. [FILL: model and provider of that chat.]
