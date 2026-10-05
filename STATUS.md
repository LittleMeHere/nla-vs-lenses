# nla-vs-lenses

**Question (Neel's MATS 12 doc):** "What can NLAs capture that Oracle Lens / J Lens don't? Is it really the stuff not in the workspace?"

**What we test:** the NLA and the oracle lens are Qwen3.6-27B itself with a LoRA, trained to describe an injected activation. When one of them names a hidden reasoning step, did it read the step from the activation, or re-solve the question from the prompt it decodes? J-Lens is a fixed linear map and cannot re-solve.

Updated 2026-10-04 (after CW-11); the pilot numbers were recomputed from the raw files on 2026-10-04. Everything here is exploratory: one model, 50–100 items per experiment, dev items only.

## Plain summary

Example used throughout: the prompt "The chemical symbol for the element with atomic number 26 is". To answer "Fe" the model has to get to "iron". "Iron" is the hidden step. J-Lens shows ten words for an activation; the oracle lens and the NLA (the "writers") each write a few sentences about it.

| # | What was done | What came out | What it means | How firm |
|---|---|---|---|---|
| 1 | Asked all three for the hidden step on 100 prompts, layer 42 | Oracle lens 80, NLA 75, J-Lens 39 | The writers mention the hidden step far more often than J-Lens's ten words do | Firm as a count of mentions (word match, checked by hand on 30) |
| 2 | Looked at J-Lens on other layers for the 36 it missed | It shows the step at another layer in 22, only after layer 42 in 14 of those | Some of the gap is which layer J-Lens is read at; it could also be the writers picking the step up before J-Lens's top ten does | Count firm, reading open |
| 3 | Sorted what the NLA wrote on 100 prompts | About 60% prompt wording or format; 66 of 98 write-ups include something the prompt rules out | The NLA pads a lot; "false" was judged by a model and not checked by hand | Padding firm, "false" unvalidated |
| 4 | Deleted the hidden step's J-Lens directions; separately deleted the prompt words' directions (CW-11) | Naming falls (oracle 65% → 37%, NLA 65% → 53%); prompt words are still reported (54% → 45%, 58% → 53%) | The writers read those directions in part. Deleting a few directions removes under 1% of the activation, so what survives cannot show the step was worked out from the prompt | Firm |
| 5 | Split each natural activation into J-Lens's top 1,024 directions (the J part, 16% of the activation) and the rest (84%), and gave each part to a reader alone (CW-12, 30 prompts) | Names the bridge: whole 45% (oracle) / 43% (NLA); J part 48% / 42%; the rest 20% / 12%. Removing a random part of the same size instead leaves 45% / 38%. PCA's top 1,024 directions (91% of the activation) give 58% / 45%. A linear probe decodes simple prompt facts from either part equally well | Both readers get the bridge from the part J-Lens reads. The rest still holds information, but the readers use it little. No sign the NLA reads the bridge from outside the J part | Fair: 30 prompts, 2 samples, word match |

## Current conclusion

Whether the prose readers re-solve depends on the task.

- **multihop (one-line trivia):** they name the hidden step about twice as often as J-Lens at L42, and they also name it at L28, where J-Lens shows it in 1/50. Deleting the step's J-Lens directions lowers naming (oracle lens 65% → 37% of samples, NLA 65% → 53%), so both read those directions in part, the oracle lens more. What survives deletion is **not** good evidence of re-solving: CW-11 shows that deleting the J-Lens directions of the prompt's own words barely stops the readers reporting those words (CW-11 below). "Mostly re-solving" is withdrawn; how much is re-solved is open.
- **chain (arithmetic), at L56 where J-Lens shows the number:** the oracle lens names the number in 36/60 (chance 12), in 31 of those without restating the starting number. Reading. The NLA is near chance.
- **brew (10-rule colour table), at ":" L42:** both readers rarely state the hidden colour in its role (2–5 of 50), and those mentions go to about zero when its directions are deleted. No sign of re-solving.
- **NLA text:** about 60% of its concepts on multihop are prompt wording or format; 66/98 readouts contain at least one claim the prompt rules out. On no task does the NLA find a hidden step the oracle lens misses.

The explanation "a reader re-solves when it can rebuild the prompt" has one weak support (CW-10: NLA readouts that recover more of the prompt name the step more often) and no causal test: CW-11's cue removal did not hide the prompt, so it could not test it.

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
| 5 | Swap: delete own step, add another item's step | 50 multihop | 42 | planted 14 | own 20, planted 7 | own 20, planted 6 | [abl3](runs/abl3/) |
| 6 | chain at the readers' training layer | 60 chain | 42 | 6 (chance 4) | 27 (chance 21) | 9 (chance 12) | [chain60](runs/chain60/) |
| 7 | chain at L56, where J-Lens shows the number | 60 chain | 56 | 25 (chance 7) | 36 (chance 12) | 13 (chance 9) | [chain60_L56](runs/chain60_L56/) |
| 8 | brew, z in its correct role (Claude / Gemini judge) | 50 brew | 42 | 0 in top-10 | 2 / 5 | 5 / 3 | [brew50](runs/brew50/) |
| 9 | brew, z in any role: random control → z deleted (Gemini judge) | 50 brew | 42 | — | 7 → 0 | 12 → 2 | [brew50_abl](runs/brew50_abl/) |

### multihop (1–5)

- **Measured.** At L42 the prose readers name the bridge in 75–80/100, J-Lens top-10 in 39; J-Lens over 11 layers reaches 76. Of the 100 items, all three readers name it in 36, only the two prose readers in 36, none in 16. At L28 the prose readers name it in 26–27/50, J-Lens in 1. After deleting ~31 bridge-neighbourhood directions, J-Lens names it in 0 and the readers still do in 20 and 23 (control 33 and 34). After the swap they keep their own bridge in 20 and name the planted one in 6–7.
- **Taken to mean.** Part of what the readers say is read (the drop under deletion; the few that follow the swap). Most of the rest is re-solved: readouts at early layers restate the question and answer it (e.g. `bf-moonwalk-firstname`, L16: "…的名字是 Neil Armstrong. This is a known fact.").
- **Not ruled out.** (a) The bridge may persist in directions outside the J-Lens neighbourhood that was deleted, which would make "re-solved" an overstatement; no probe-based presence measure yet. (b) The planted step is weak: J-Lens shows it in only 14/50 even at 16× scale, so the swap is a weak test of reading. (c) Word match counts mentions, which can overstate role-correct claims.
- An earlier, coarser version of 4–5 (cos > 0.5, ~1,150 directions) damaged the random control too; superseded. Raw in [abl2](runs/abl2/).

### chain (6–7)

- **Measured.** J-Lens finds an intermediate at 0 items for L20–36, 6–8 for L40–52, 25 at L56, 6 at L60. At L42 no reader is clearly above chance. At L56 the oracle lens names an intermediate in 36/60 (chance 12.1), the NLA in 13 (chance 8.9). The oracle lens states the start number in 12 readouts; 31 of its 36 hits do not contain it.
- **Taken to mean.** Where the number is linearly present, the oracle lens reads it, more often than J-Lens top-10, and without the context it would need to recompute. Random example (`chain-d2p-0298`, start 24, halve, add 6 → 12): "12. A quick nod, a flushed face, and a wide smile. The party…".
- **Not ruled out.** Readers are off their training layer at L56. Readouts contain many numbers (oracle 6–9, NLA 4–5 distinct per readout), hence the high chance rates. Predictions were written before the L42 run; the L56 run was added after seeing that J-Lens peaks there.

### brew (8–9)

- **Measured.** Read at ":" after "Answer" for all 50 items. J-Lens top-10 contains z at L42 in 0/50, L56 19, L60 46 (answer 50 at L56 and L60). Role-correct z in the originals: NLA 5 (Claude judge) / 3 (Gemini), oracle 2 / 5; chance about 1. The two judges give the same intermediate label on 34/50 readouts for each reader. Deleting z's directions (4.5 on average): z in any role, oracle 7 → 0, NLA 12 → 2 (Gemini judge). Swap rows are single digits and not interpretable.
- **Taken to mean.** On brew the readers rarely claim the hidden step, and what they do mention is removed by the deletion. No re-solving here.
- **Not ruled out.** Small counts; judge disagreement of about a third means single-digit differences are noise. The judge is not validated against human labels.

### What the NLA's text consists of (multihop, 100 readouts)

1,696 concepts (split by `claude -p`, bucketed by word match, `scripts/classify_claims.py`): hidden step or answer 12%; also in J-Lens top-10 15%; prompt words 22%; format description 22%; next-token 1%; other 27%. The "other" 461, judged against the prompt (`scripts/judge_other.py`): form 45%, unstated and not ruled out 27% (mostly loose associations), contradicted by the prompt 22%, restates 6%. 66/98 readouts have at least one contradicted concept. NLA readouts average 127 words and about 7 quoted strings; oracle-lens readouts 73 words and under 1.

### CW-10: J-Lens at other layers, and prompt recovery (2026-10-04, exploratory, no model calls)

Design: `designs/CW-10_prompt_recovery_and_j_any_layer.md`. Code: `scripts/cw10.py`. Output: `runs/cw10/analysis.txt`. Uses the saved pilot readouts (1 NLA sample, 1 oracle readout per cell, word match). The script first reproduces CW-1: J-Lens 39, oracle 80, NLA 75, J-Lens at any of 11 layers 76.

**J-Lens at other layers.** Of the 36 items where the oracle lens and the NLA name the bridge at L42 and J-Lens does not:
- J-Lens top-10 names the bridge at one or more of layers 20–60 (step 4) in 22, and at no layer in 14.
- First layer for the 22: before L42 in 8, after L42 in 14 (10 of them first at L52 or later).
- For the 16 items no reader names at L42, J-Lens names the bridge later (L52–60) in 11.

**Prompt recovery.** Score = share of the prompt's content words (bridge and answer words excluded; mean 6 words per prompt) found in the readout. Scored against another item's prompt it is 0.11–0.18; against the right prompt 0.44–0.66.

| readouts | n (names bridge) | oracle AUC | NLA AUC | NLA, length only |
|---|---|---|---|---|
| after removal (CW-4), the pre-stated test | 50 (20 / 23) | 0.58 [0.41, 0.75] | 0.62 [0.45, 0.78] | 0.37 |
| intact L42 | 100 (80 / 75) | 0.62 [0.47, 0.76] | 0.71 [0.60, 0.81] | 0.42 |
| L28 | 50 (26 / 27) | 0.47 [0.31, 0.64] | 0.73 [0.59, 0.86] | 0.50 |
| L36 | 50 (33 / 28) | 0.65 [0.49, 0.80] | 0.67 [0.51, 0.82] | 0.41 |
| L16 | 50 (9 / 11) | 0.62 [0.39, 0.83] | 0.67 [0.49, 0.84] | 0.45 |

AUC = how well the recovery score separates readouts that name the bridge from those that do not; intervals are bootstrap over items.

**Taken to mean.**
- For 14 of the 36 prose-only items J-Lens never shows the bridge at any layer; these are the strongest re-solving candidates. For 14 others J-Lens shows it only after L42, so at L42 the prose readers name a step that J-Lens's top-10 shows only later. That fits re-solving, and it also fits content present at L42 below J-Lens's top-10.
- NLA readouts that recover more of the prompt name the bridge more often (0.71 at L42, 0.73 at L28), and readout length does not explain it. The oracle lens shows no clear relation.
- The pre-stated test (after removal) is inconclusive at 50 items, as the design expected: both intervals include 0.5, and the 0.65 hoped for is not reached at the point estimate.
- **Word match checked by hand.** 30 random NLA L42 readouts (seed 0) were labelled by hand by the author, blind, for whether the readout mentions the bridge: word match agrees on 29 of 30 (it misses one plural, "kidneys"). Claude's independent reading matches the author's on 30 of 30. `runs/cw10/handcheck_nla30.json`.
- **Most likely objection:** a readout that reads the activation well would contain both prompt words and the bridge, so the relation does not by itself show re-solving. Not addressed. Sixteen AUCs were computed; treat single cells as exploratory.

### CW-11: cue removal, 4 samples per readout (2026-10-04)

Design, written and committed before the run: `designs/CW-11_cue_removal.md` (the `cue_orth` condition was added after the J-Lens check and before any reader output was read). Code: `scripts/cw11_pass_a.py`, `cw11c_pass_a.py`, `cw11_pass_b.py`, `cw11_analyze.py`. Raw: `runs/cw11/`. 50 multihop items (those of CW-4), L42, oracle lens and NLA at 4 samples per cell with fixed seeds, word match, bf16. NLA `ceselder/qwen3.6-27b-nla-rl`, `iter_000400`.

**Conditions.** intact; rand (106 random vocabulary directions removed, the mean size of the cue set); bridge (the bridge's J-Lens neighbourhood, mean 31 directions, as CW-4); cue (the J-Lens neighbourhoods of the prompt's content words, mean 106 directions); cue_orth (the same, made orthogonal to the bridge set first, so the bridge's J-Lens score is unchanged: 1.95 in both); both.

**Measured** (share of the 200 samples that name the bridge; prompt recovery = share of the prompt's content words in the readout):

| condition | oracle: names bridge | oracle: recovery | NLA: names bridge | NLA: recovery | cue words in J-Lens top-50 |
|---|---|---|---|---|
| intact | 0.67 | 0.58 | 0.64 | 0.59 | 6.7 |
| rand | 0.65 | 0.54 | 0.65 | 0.58 | 7.0 |
| cue | 0.64 | 0.45 | 0.60 | 0.53 | 0.02 |
| cue_orth | 0.71 | 0.51 | 0.72 | 0.55 | 0.28 |
| bridge | 0.37 | 0.51 | 0.53 | 0.56 | 6.3 |
| both | 0.40 | 0.45 | 0.54 | 0.50 | 0.02 |

Paired differences over the 50 items, 95% bootstrap:
- bridge − rand, naming: oracle −0.29 [−0.40, −0.18]; NLA −0.12 [−0.20, −0.06].
- cue_orth − rand, naming (the main contrast): oracle +0.06 [0.00, +0.12]; NLA +0.08 [+0.01, +0.15].
- cue − rand, naming: oracle −0.02 [−0.09, +0.05]; NLA −0.05 [−0.14, +0.04].
- both − bridge, naming: oracle +0.03 [−0.04, +0.10]; NLA +0.01 [−0.06, +0.08].
- Recovery, cue − rand: oracle −0.09 [−0.12, −0.06]; NLA −0.05 [−0.09, −0.02]. cue_orth − rand: −0.03 and −0.03.
- Under cue_orth, J-Lens top-10 shows the bridge in 33/50 (14 intact): removing the cue words raises the bridge's rank.
- All 4 samples agree on naming in 36/50 (oracle) and 39/50 (NLA) intact items.

**Taken to mean.**
- **Deleting a word's J-Lens directions does not hide it from the prose readers.** The cue words vanish from J-Lens (6.7 → 0.02 per item in its top-50), yet the readouts still contain them at 0.45–0.53 against 0.54–0.58 under random removal (0.11–0.18 for another item's prompt). These words are in the prompt, so there is nothing to re-solve. The readers get them from what is left of the activation.
- So a bridge mention that survives J-Lens deletion (CW-3, CW-4, and here) cannot be counted as re-solved. This is objection (a) of the multihop section, now shown directly.
- The bridge's J-Lens directions do matter: deleting them lowers naming by 29 points for the oracle lens and 12 for the NLA, far more than it lowers prompt recovery. The oracle lens leans on them more.
- **The planned test of re-solving is uninformative**, by the rule set before the run: cue removal lowered prompt recovery by only 3 to 9 points, so the prompt was not hidden. Naming did not fall under either cue removal, and rose slightly under cue_orth, where the bridge becomes more prominent in J-Lens.
- **Most likely objection:** the readers may re-derive both the cue words and the bridge from other content left in the activation, which would also produce this pattern. Not addressed: it needs a removal that takes the information out wherever it sits (reader-free erasure), or the planted-concept test.
- n: 50 items, 4 samples each, one prompt family, one model, word match (checked by hand on 30 readouts: 29 agree).

### CW-12: what each reader gets from the J part and from the rest (2026-10-05)

Design, committed before the run: `designs/CW-12_j_split.md`. Code: `jobs/cw12_a_sweep.py`, `jobs/cw12_b_split.py`, `jobs_nla/cw12_c_nla.py`, `jobs_pca/cw12_d_pca.py`, `jobs_nla2/cw12_e_nla.py`, `scripts/worker.py`, `scripts/cw12_analyze.py`, `scripts/cw12_probe.py`, `scripts/cw12_fig.py`. Raw: `runs/cw12/`. Figure: `figs/cw12_split_final_30items.png`.

**Setup.** Qwen3.6-27B, L42, the first 30 of the 50 multihop items of CW-4, same saved activations (bf16 capture). J part = projection onto the top 1,024 principal directions of all unit-normalised J-Lens token directions (W_U[t] J) at L42; these hold 93% of the spread of those directions. The basis is in the public repo (`jbasis/`). k = 1,024 was chosen from a 5-item check (`runs/cw12/sweep.jsonl`: at k = 64 and 256 J-Lens still reads its tokens from the rest; at 1,024 it does not). Controls: a random 1,024-dimensional subspace (seed 0), and, added after the first results, the top 1,024 PCA directions of 16,000 L42 activations from benchmark prompts (all token positions; multihop, association, basic readout, multilingual, chain, brew), centred. Readers: oracle lens and NLA (`nla-rl`, `iter_000400`), 2 samples per cell, fixed seeds; both rescale their input. Word match.

**Measured** (share of the 60 samples per cell that name the bridge, 95% bootstrap over items; prompt recovery as CW-10):

| part given to the reader | share of squared norm | oracle: bridge | NLA: bridge | oracle: recovery | NLA: recovery |
|---|---|---|---|---|
| whole | 1.00 | 0.45 [0.30, 0.60] | 0.43 [0.28, 0.58] | 0.57 | 0.57 |
| J part | 0.16 | 0.48 [0.32, 0.65] | 0.42 [0.27, 0.58] | 0.49 | 0.50 |
| the rest | 0.84 | 0.20 [0.10, 0.32] | 0.12 [0.03, 0.23] | 0.12 | 0.25 |
| random 1,024 | 0.20 | 0.17 [0.05, 0.30] | 0.35 [0.20, 0.50] | 0.27 | 0.33 |
| all but random | 0.80 | 0.45 [0.28, 0.62] | 0.38 [0.23, 0.53] | 0.51 | 0.53 |
| PCA top 1,024 | 0.91 | 0.58 [0.42, 0.73] | 0.45 [0.28, 0.62] | 0.57 | 0.55 |
| all but PCA | 0.10 | 0.13 [0.05, 0.25] | 0.05 [0.00, 0.12] | 0.03 | 0.06 |

Paired differences in naming the bridge, over the 30 items:
- J part − the rest: oracle +0.28 [0.12, 0.47]; NLA +0.30 [0.12, 0.48].
- J part − whole: oracle +0.03 [−0.07, 0.15]; NLA −0.02 [−0.15, 0.13].
- the rest − all but random (removing the J part against removing a random part): oracle −0.25 [−0.42, −0.10]; NLA −0.27 [−0.43, −0.10].
- J part − random 1,024: oracle +0.32 [0.13, 0.50]; NLA +0.07 [−0.08, 0.22].
- J part − PCA top 1,024: oracle −0.10 [−0.23, 0.02]; NLA −0.03 [−0.20, 0.13].
- J-Lens top-10 names the bridge in 4/30 on the whole activation, 5/30 on the J part, 0/30 on the rest.
- J and PCA subspaces overlap 0.31 (0.20 for unrelated subspaces); 30% of the centred activation variance lies in the J part.
- Probe (no GPU, 50 items, leave-one-out logistic regression, `runs/cw12/probe.txt`): "answer is a number" and "prompt contains number / country / month" decode at AUC 0.93–1.00 from the whole activation, the J part, the rest and a random part alike; shuffled labels 0.37–0.51.

**Taken to mean.**
- For both readers the J part is sufficient: 16% of the activation gives as much as all of it.
- Removing the J part costs both readers about 25 points more than removing a random part of the same size. This is the evidence that J-Lens's directions matter to them specifically.
- Sufficiency is not specific to J-Lens. The high-variance PCA part is sufficient too (it is 91% of the activation), and for the NLA a random fifth is nearly sufficient (0.35 against 0.42). The oracle lens needs the J part intact; the NLA copes with a thinned copy.
- The rest is not empty: simple prompt facts are linearly decodable from it. The readers get little from it.
- On natural prompts there is no sign that the NLA reads the bridge from outside the part J-Lens reads.
- **Most likely objection:** a part of an activation is off-distribution for a reader trained on whole activations, so "the reader gets little from the rest" may be a failure to parse it and not absence of use. The random control argues against this (its complement, also a part, reads fine), but the rest has a different statistical character from a random complement. Not further addressed.
- n: 30 items, 2 samples per cell, one prompt family, one layer, one definition of the J part, word match (hand-checked on other readouts). The probe labels are coarse; no probe for the bridge itself (one item per bridge).

### CW-13: the bridge's rank in J-Lens's full list, and the template lens (2026-10-05, no GPU)

Code: `scripts/cw13_jrank.py`, `scripts/cw13_template.py`, `scripts/cw13_fig.py`. Output: `runs/jrank/`. Figures: `figs/cw13_bridge_rank.png`, `figs/cw13_where_is_the_bridge.png`. Not pre-registered; both analyses were chosen after CW-12's results were known.

**Setup.** The 50 multihop items with saved L42 activations (CW-4), and the 30 split into parts in CW-12. J-Lens scores for all 248,320 tokens, computed on CPU from the neuronpedia map and the model's output-word matrix (same cosine readout as the benchmark). The bridge's rank is the best rank among tokens whose text equals a bank intermediate (letters and digits, case-insensitive); 46 of 50 items have such a token. Template lens: `camilablank/workspace-lenses` @ d740106d, 13,174 words, layer 42, plain projection of the activation (minus a mean over 16,000 benchmark-prompt positions for whole activations; raw for parts). The file does not include the lens's own decode rule, so this projection is an assumption. The bridge is in its vocabulary for 41 of 50 items (25 of the 30).

**Measured.**
- J-Lens, 50 items: bridge within the top 10 in 15 (30%), top 50 in 24 (48%), top 200 in 35 (70%), top 1,000 in 38. Median rank 47.
- Template lens, 50 items: top 10 in 12, top 50 in 25, top 200 in 31. Median rank 34.
- For comparison, on the same 50 items the oracle lens names the bridge in 66% of single readouts and the NLA in 64% (CW-11, intact, 4 samples).
- By part (30 items), bridge within the top 50: J-Lens 10 whole, 10 J part, 0 the rest (median rank in the rest 77,789). Template lens (25 items) 14 whole, 12 J part, 11 the rest (median ranks 37, 61, 93); random 1,024 directions 5, all but those 10; PCA top 1,024 14, all but those 4.

**Taken to mean.**
- "The prose readers name the bridge about twice as often as J-Lens" (results 1 and CW-1) is largely the top-10 cutoff. Looking 200 words down J-Lens's list gives the bridge as often as one prose readout does.
- The bridge is still present in the rest of the activation: the template lens finds it there nearly as well as in the J part. J-Lens finding nothing in the rest is by construction. So in CW-12 the prose readers get little of the bridge from the rest although it is there to be read.
- For Neel's question: there is bridge content outside the J part, and the NLA does not pick it up better than the oracle lens (12% against 20%).
- **Most likely objection:** the template lens was applied with an assumed decode rule, and to parts without a matched mean. Not addressed; ask the lens's author. Also 25 to 30 items.

## Limits

- One model. One sample per NLA cell. Dev items only; nothing held out.
- multihop scoring is word match. Judges are unvalidated.
- No reader-free label for whether the model uses the hidden step.
- No SAE: no labelled SAE exists for Qwen3.6-27B (Gemma 3 has Gemma Scope 2).

## Next

1. Done as CW-10, CW-11 and CW-12. Still open: a removal that does not depend on J-Lens (reader-free erasure of the bridge, fitted on other items), with a probe to confirm nothing is left, then the same readers at 4 samples. That is what can separate read from re-solved on multihop.
2. Planted concepts split into J-space and non-J parts (as Viswanath did on Llama-3.3-70B), read by all three readers. This tests "outside the workspace" directly.
3. Role-aware scoring of the multihop readouts with a validated role judge.
4. A probe-based presence measure for the bridge per layer.

## Prior work

- Viswanath, "Models are blind outside the J-space. NLAs aren't." (LessWrong, 2026-07-08): on Llama-3.3-70B, an injected concept's non-J part is invisible to the model and read by the NLA. Injection setup; does not cover natural prompts or re-solving.
- Bhatia, Blank, Nanda, "Towards surfacing model algorithms with meta-tokens in the J-space" (LessWrong, 2026-07-20): J-Lens readouts on Qwen3.6-27B carry process tokens. Context for J-Lens showing task words, not the hidden step, at mid layers.
- Prabhu, "Can you hide from a natural language autoencoder?" (LessWrong, 2026-06-24): NLA explanations on Qwen2.5-7B can be flipped while behaviour is unchanged.
- WorkspaceBench post (Nanda et al.): single-token lenses are bag-of-words; NLAs surface workspace content and hallucinate; oracle lens more trustworthy.

## Code

[scripts/](scripts/): `smoke_pass_a.py` and `smoke_pass_b.py` (lenses and oracle lens, then NLA); `ablate_pass.py`, `ablate2_pass.py` (deletion and swap); `chain_pass.py` (last-token reads for chain and brew); `classify_claims.py`, `judge_other.py`, `judge_roles.py`, `score_chain.py` (scoring); `patch_jlens_mem.py` (memory fixes to the WorkspaceBench J-Lens and R-lens code; scores are bf16 after it); `pod_bootstrap.sh`, `pod_sync.sh`, `*_chain.sh` (running on a rented GPU).
