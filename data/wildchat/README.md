# WildChat deployment transcripts (realism win rate haystack)

The 20 WildChat conversations used as deployment ("haystack") transcripts for the paper's
realism win rate measure (paper appendix "Deployment transcripts"):

- `wildchat_harmful` — 10 conversations involving misuse from the user.
- `wildchat_vulnerable` — 10 conversations where the user is in an unusual or vulnerable
  situation.
- `wildchat_mixed` — the exact concatenation of the two files above (20 rows); this is the
  WildChat portion of the haystack as consumed by `code/realism_win_rate.py`
  (`--haystack_path`).

Files are byte-for-byte copies of `dataset/realism_win_rate/` in the experiment fork,
JSONL with one conversation per row. Each row keeps the original WildChat record: the
conversation `hash` (the stable identifier in the WildChat distribution), model, language,
turn count, the `classification` from the selection step, and the full `conversation` array
as published in WildChat (including WildChat's own `hashed_ip`/header metadata and redaction
flags — no fields were added or removed here).

## Provenance and license

The underlying conversations are from **WildChat** (Zhao et al., 2024,
[allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)), distributed
under the **ODC-BY** license; this subset is redistributed with attribution accordingly.
The harmful/vulnerable subset selection follows Kissane, MacDiarmid and Roger (2026),
*Measuring and improving coding audit realism with deployment resources*, Anthropic
Alignment Science Blog, https://alignment.anthropic.com/2026/coding-audit-realism/.

The remainder of the paper's haystack — 20 Claude Code transcripts donated by Cambridge
Boston Alignment Initiative research fellows — was donated for internal use only and is
**not** included (see the paper's "Transcript selection" appendix).
