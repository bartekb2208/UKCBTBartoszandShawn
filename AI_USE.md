# AI use disclosure

No AI model is part of the prediction pipeline. The predictions come only from `src/` (feature code plus LightGBM) run on organiser data.

AI assistance was used to plan and write the code.

## Assistant used for the build

- **Model:** Claude Opus 5.5 (`claude-opus-5-5`)
- **Provider:** Anthropic, through Claude Code (VS Code extension), on 24 October 2026
- **Settings:** Claude Code defaults; auto permission mode; no fine-tuning, no system prompt beyond Claude Code's own
- **What it did:** read the organiser README, starter notebook and `payments.py`; profiled the training data (schemas, counts, label breakdowns, and individual example scenarios); wrote all code in `src/`, `run.sh`, this file and `NOTES.md`; ran the cross-validation experiments and chose features from fold scores.
- **Data shown to the assistant:** organiser training and validation data only (rows of `train/` and `validation/`, and `train/labels.csv`), read locally. No test labels exist on our side, and no external data was used.
- **Prompts:** the team gave the assistant the plan files `PLAN.md` and `PRIMER.md` (written with AI help in an earlier planning chat), then the instruction: *"Look through the entire repo, read everything especially the plan and primer, and execute. The objective variable we're trying to optimise is the score output."* Later prompts are listed below as they happen.

## Earlier planning chat

`PLAN.md` and `PRIMER.md` were drafted in a separate AI planning chat before the build. Team to fill in: model, provider and settings used there.

## Later prompts

- (add each follow-up instruction here)
