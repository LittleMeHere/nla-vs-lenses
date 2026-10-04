# nla-vs-lenses

**Question (Neel's MATS 12 doc):** "What can NLAs capture that Oracle Lens / J Lens don't? Is it really the stuff not in the workspace?"

**What we test:** the NLA and the oracle lens are Qwen3.6-27B itself with a LoRA, trained to describe an injected activation. When one of them names a hidden reasoning step, did it read the step from the activation, or re-solve the question from the prompt it decodes? J-Lens is a fixed linear map and cannot re-solve.

Updated 2026-10-03; every number below was recomputed from the raw files on 2026-10-04. Everything here is exploratory: one model, 50–100 items per experiment, dev items only.

## Current conclusion

Whether the prose readers re-solve depends on the task.

- **multihop (one-line trivia):** they name the hidden step about twice as often as J-Lens at L42, but they also name it at L28, where J-Lens shows it in 1/50, and about two-thirds of their mentions survive deleting the step's directions from the activation. Mostly re-solving.
- **chain (arithmetic), at L56 where J-Lens shows the number:** the oracle lens names the number in 36/60 (chance 12), in 31 of those without restating the starting number. Reading. The NLA is near chance.
- **brew (10-rule colour table), at ":" L42:** both readers rarely state the hidden colour in its role (2–5 of 50), and those mentions go to about zero when its directions are deleted. No sign of re-solving.
- **NLA text:** about 60% of its concepts on multihop are prompt wording or format; 66/98 readouts contain at least one claim the prompt rules out. On no task does the NLA find a hidden step the oracle lens misses.

Untested explanation: a reader re-solves when it can rebuild the prompt from the activation (a one-line question, not a rule table).

Not tested: content known to be outside J-space. So "is it the stuff not in the workspace?" is not answered directly; see Next.

## Setup

- **Model:** Qwen/Qwen3.6-27B, bf16, one 80 GB GPU.
- **Prompts:** WorkspaceBench banks ([repo](https://github.com/camilablank/workspace-bench)), unmodified.
  - `multihop` (100 items): "Fact: The chemical symbol for the element with atomic number 26 is" → hidden step *iron*, answer *Fe*.
  - `chain_intermediates` (first 60): "Halve it, rounding down." ×3, "The starting number is 23" → hidden steps 11, 5; answer 2.
  - `brew_intermediates` (first 50): ten colour rules, two stirs, start colour last → hidden step = colour after the first stir.
- **Method:** run the prompt, take the residual-stream vector at the bank's read token and one layer, give the same vector to each reader. Edits (deletion, swap) are applied to that vector before the readers see it.
- **Readers:**
  - J-Lens: top-10 tokens, cosine readout ([neuronpedia/jacobian-lens](https://huggingface.co/neuronpedia/jacobian-lens), Qwen3.6-27B n1000 wikitext). Also R-lens and logit lens.
  - Oracle lens: WorkspaceBench default LoRA (`agu18dec/local-workspace`, `s3d.ddp600.s0/iter_000600`), one readout of four bullets.
  - NLA: [ceselder/qwen3.6-27b-nla-rl](https://huggingface.co/ceselder/qwen3.6-27b-nla-rl), adapter `iter_000400`, trained at L42, one sample.
- **Scoring:**
  - multihop: exact word match of the bank's intermediate in the readout. This counts mentions, not role-correct claims.
  - chain: the number as digits, an English number word or a Chinese numeral (`scripts/score_chain.py`).
  - brew: a prompt-blind role judge (`scripts/judge_roles.py`) that sees the readout and the candidate colours and labels the start, intermediate and answer roles. Run with Claude and with Gemini 3.8 Flash; not validated against human labels.
  - Chance: the same readout scored against other items' hidden steps (equivalent to giving the reader another item's activation).

## Results

Counts are items whose readout names the hidden step.

| # | Experiment | Items | Layer | J-Lens | Oracle lens | NLA | Raw |
|---|---|---|---|---|---|---|---|
| 1 | Main comparison | 100 multihop | 42 | 39 | 80 | 75 | [multihop100](runs/multihop100/) |
|   | chance | | | 0.7 | 1.7 | 1.2 | |
| 2 | Earlier layers | 50 multihop | 16 / 28 / 36 | 0 / 1 / 9 | 9 / 26 / 33 | 11 / 27 / 28 | [sweep50](runs/sweep50/) |
| 3 | Delete the step's J-Lens directions (2–6 per item); random-direction control unchanged | 50 multihop | 42 | 15 → 0 | 30 → 17 | 36 → 25 | [abl50](runs/abl50/) |
| 4 | Delete the step's neighbourhood (cos > 0.8, ~31 directions). Control = same number of random directions | 50 multihop | 42 | 15 → 0 | 33 → 20 | 34 → 23 | [abl3](runs/abl3/) |
| 5 | Swap: delete own step, add another item's step | 50 multihop | 42 | planted 14 | own 20, planted 7 | own 20, planted 6 | [abl3](runs/abl3/) |
| 6 | chain at the readers' training layer | 60 chain | 42 | 6 (chance 4) | 27 (chance 21) | 9 (chance 12) | [chain60](runs/chain60/) |
| 7 | chain at L56, where J-Lens shows the number | 60 chain | 56 | 25 (chance 7) | 36 (chance 12) | 13 (chance 9) | [chain60_L56](runs/chain60_L56/) |
| 8 | brew, z in its correct role (Claude / Gemini judge) | 50 brew | 42 | 0 in top-10 | 2 / 5 | 5 / 3 | [brew50](runs/brew50/) |
| 9 | brew, z in any role: random control → z deleted (Gemini judge) | 50 brew | 42 | — | 7 → 0 | 12 → 2 | [brew50_abl](runs/brew50_abl/) |

### multihop (1–5)

- **Measured.** At L42 the prose readers name the bridge in 75–80/100, J-Lens top-10 in 39; J-Lens over 11 layers reaches 76. Of the 100 items, all three readers name it in 36, only the two prose readers in 36, none in 16. At L28 the prose readers name it in 26–27/50, J-Lens in 1. After deleting ~31 bridge-neighbourhood directions, J-Lens names it in 0 and the readers still do in 20 and 23 (control 33 and 34). After the swap they keep their own bridge in 20 and name the planted one in 6–7.
- **Taken to mean.** Part of what the readers say is read (the drop under deletion; the few that follow the swap). Most of the rest is re-solved: readouts at early layers restate the question and answer it (e.g. `bf-moonwalk-firstname`, L16: "…的名字是 Neil Armstrong. This is a known fact.").
- **Not ruled out.** (a) The bridge may persist in directions outside the J-Lens neighbourhood that was deleted, which would make "re-solved" an overstatement; no probe-based presence measure yet. (b) The planted step is weak: J-Lens shows it in only 14/50 even at 16× scale, so the swap is a weak test of reading. (c) Word match counts mentions; role-aware scoring on brew (below) shows mentions overstate claims.
- An earlier, coarser version of 4–5 (cos > 0.5, ~1,150 directions) damaged the random control too; superseded. Raw in [abl2](runs/abl2/).

### chain (6–7)

- **Measured.** J-Lens finds an intermediate at 0 items for L20–36, 6–8 for L40–52, 25 at L56, 6 at L60. At L42 no reader is clearly above chance. At L56 the oracle lens names an intermediate in 36/60 (chance 12.1), the NLA in 13 (chance 8.9). The oracle lens states the start number in 12 readouts; 31 of its 36 hits do not contain it.
- **Taken to mean.** Where the number is linearly present, the oracle lens reads it, more often than J-Lens top-10, and without the context it would need to recompute. Random example (`chain-d2p-0298`, start 24, halve, add 6 → 12): "12. A quick nod, a flushed face, and a wide smile. The party…".
- **Not ruled out.** Readers are off their training layer at L56. Readouts contain many numbers (oracle 6–9, NLA 4–5 distinct per readout), hence the high chance rates. Predictions were written before the L42 run; the L56 run was added after seeing that J-Lens peaks there.

### brew (8–9)

- **Measured.** Read at ":" after "Answer" for all 50 items. J-Lens top-10 contains z at L42 in 0/50, L56 19, L60 46 (answer 50 at L56 and L60). Role-correct z in the originals: NLA 5 (Claude judge) / 3 (Gemini), oracle 2 / 5; chance about 1. The two judges give the same intermediate label on 34/50 readouts for each reader. Deleting z's directions (4.5 on average): z in any role, oracle 7 → 0, NLA 12 → 2 (Gemini judge). Swap rows are single digits and not interpretable.
- **Taken to mean.** On brew the readers rarely claim the hidden step, and what they do mention is removed by the deletion. No re-solving here.
- **Not ruled out.** Small counts; judge disagreement of about a third means single-digit differences are noise. The judge prompt is not validated against human labels.

### What the NLA's text consists of (multihop, 100 readouts)

1,696 concepts (split by `claude -p`, bucketed by word match, `scripts/classify_claims.py`): hidden step or answer 12%; also in J-Lens top-10 15%; prompt words 22%; format description 22%; next-token 1%; other 27%. The "other" 461, judged against the prompt (`scripts/judge_other.py`): form 45%, unstated and not ruled out 27% (mostly loose associations), contradicted by the prompt 22%, restates 6%. 66/98 readouts have at least one contradicted concept. NLA readouts average 127 words and about 7 quoted strings; oracle-lens readouts 73 words and under 1.

## Limits

- One model. One sample per NLA cell. Dev items only; nothing held out.
- multihop scoring is word match. Judges are unvalidated.
- No reader-free label for whether the model uses the hidden step.
- No SAE: no labelled SAE exists for Qwen3.6-27B (Gemma 3 has Gemma Scope 2).

## Next

1. Test the explanation: per item, how much of the prompt does each readout recover, and does that predict naming the hidden step? No GPU needed.
2. Inject a concept split into J-space and non-J parts (as Viswanath did on Llama-3.3-70B) and ask what each of the three readers reports. This tests "outside the workspace" directly.
3. Role-aware scoring of the multihop readouts with a validated judge.
4. A probe-based presence measure for the bridge per layer.


## Prior work

- Viswanath, "Models are blind outside the J-space. NLAs aren't." (LessWrong, 2026-07-08): on Llama-3.3-70B, an injected concept's non-J part is invisible to the model and read by the NLA. Injection setup; does not cover natural prompts or re-solving.
- Bhatia, Blank, Nanda, "Towards surfacing model algorithms with meta-tokens in the J-space" (LessWrong, 2026-07-20): J-Lens readouts on Qwen3.6-27B carry process tokens. Context for J-Lens showing task words, not the hidden step, at mid layers.
- Prabhu, "Can you hide from a natural language autoencoder?" (LessWrong, 2026-06-24): NLA explanations on Qwen2.5-7B can be flipped while behaviour is unchanged.
- WorkspaceBench post (Nanda et al.): single-token lenses are bag-of-words; NLAs surface workspace content and hallucinate; oracle lens more trustworthy.

## Code

[scripts/](scripts/): `smoke_pass_a.py` and `smoke_pass_b.py` (lenses and oracle lens, then NLA); `ablate_pass.py`, `ablate2_pass.py` (deletion and swap); `chain_pass.py` (last-token reads for chain and brew); `classify_claims.py`, `judge_other.py`, `judge_roles.py`, `score_chain.py` (scoring); `patch_jlens_mem.py` (memory fixes to the WorkspaceBench J-Lens and R-lens code; scores are bf16 after it); `pod_bootstrap.sh`, `pod_sync.sh`, `*_chain.sh` (running on a rented GPU).
