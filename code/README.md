# Code

## `realism_win_rate.py`

The realism win rate implementation used in the paper, verbatim from the experiment fork
(`experiments/transcript_realism/realism_win_rate.py`). Each audit transcript ("needle") is
paired with deployment transcripts ("haystack") and a judge model ranks each pair from least
to most realistic using the prompt in `../prompts/realism_win_rate/realism_ranking_prompt.txt`
(inlined in the script). Win rate = fraction of pairs where the judge ranks the audit
transcript as less realistic.

Notes for running it outside the fork:

- It imports `petri.openai_safety`; a verbatim copy is included here as `openai_safety.py`
  (adjust the import or put it on `PYTHONPATH` as `petri/openai_safety.py`).
- The WildChat portion of the paper's haystack ships in `../data/wildchat/` — point
  `--haystack_path` at `data/wildchat/wildchat_mixed` (or the harmful/vulnerable split).
  The Claude Code portion (CBAI donations) is **not distributed** (internal use only); to
  fully reproduce the paper's haystack, supply your own JSONL of `{transcript, ...}` or
  WildChat-format `{conversation, ...}` entries for that part.
- API keys are read from the environment (`.env`) as in the fork.

## `openai_safety.py`

23-line helper that returns the OpenAI `safety_identifier` used for OpenAI-judge calls
(override with the `OPENAI_SAFETY_IDENTIFIER` env var).
