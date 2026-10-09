# nla-vs-lenses

**Question (Neel's MATS 12 doc):** "What can NLAs capture that Oracle Lens / J Lens don't? Is it really the stuff not in the workspace?"

**What we test:** the NLA and the oracle lens are Qwen3.6-27B itself with a LoRA, trained to describe an injected activation. When one of them names a hidden reasoning step, did it read the step from the activation, or re-solve the question from the prompt it decodes? J-Lens is a fixed linear map and cannot re-solve.

Updated 2026-10-05 (after CW-14); the pilot numbers were recomputed from the raw files on 2026-10-04. Everything here is exploratory except CW-14's three pre-set tests: one model, 30–100 items per experiment, dev items only.

## Plain summary (joint; see Collaboration and resources)

Example used throughout: the prompt "The chemical symbol for the element with atomic number 26 is". To answer "Fe" the model has to get to "iron". "Iron" is the hidden step. J-Lens shows ten words for an activation; the oracle lens and the NLA (the "writers") each write a few sentences about it.

| # | What was done | What came out | What it means | Who | How firm |
|---|---|---|---|---|---|
| 1 | Asked all three for the hidden step on 100 prompts, layer 42 | Oracle lens 80, NLA 75, J-Lens 39. Layer 42 is outside the oracle lens's trained layers; at layer 44, which is inside them, it names the step in 72 | The writers mention the hidden step far more often than J-Lens's ten words do | this repo | Firm as a count of mentions (word match, checked by hand on 30) |
| 2 | Looked at J-Lens on other layers for the 36 it missed | It shows the step at another layer in 22, only after layer 42 in 14 of those | Some of the gap is which layer J-Lens is read at; it could also be the writers picking the step up before J-Lens's top ten does | this repo | Count firm, reading open |
| 3 | Colour task: two positions where the hidden colour can be read; patching one changes the answer, the other does not | The tools report the colour about equally at both | A tool saying something does not show the model is using it | Vishesh | Firm, one task |
| 4 | Planted a known concept, split into its J part and the rest (his split: 16 directions per concept) | Both writers name "the rest" only at full strength; the oracle lens more than the NLA | No sign that either writer sees content outside J-space | Vishesh | Fair; absence of evidence |
| 5 | Sorted what the NLA wrote on 100 prompts | About 60% prompt wording or format; 66 of 98 write-ups include something the prompt rules out | The NLA pads a lot; "false" was judged by a model and not checked by hand | this repo | Padding firm, "false" unvalidated |
| 6 | Deleted the hidden step's J-Lens directions; separately deleted the prompt words' directions (CW-11) | Naming falls (oracle 65% → 37%, NLA 65% → 53%); prompt words are still reported (54% → 45%, 58% → 53%) | The writers read those directions in part. Deleting a few directions removes under 1% of the activation, so what survives cannot show the step was worked out from the prompt | this repo; both found the flaw | Firm |
| 7 | Split each natural activation into J-Lens's top 1,024 directions (the J part, 16% of the activation) and the rest (84%), and gave each part to a reader alone (CW-12, 30 prompts) | Names the bridge: whole 45% (oracle) / 43% (NLA); J part 48% / 42%; the rest 20% / 12%. Removing a random part of the same size instead leaves 45% / 38%. PCA's top 1,024 directions (91% of the activation) give 58% / 45%. A linear probe decodes simple prompt facts from either part equally well | Both readers get the bridge from the part J-Lens reads. The rest still holds information, but the readers use it little. No sign the NLA reads the bridge from outside the J part | this repo | Fair: 30 prompts, 2 samples, word match |
| 8 | Looked 200 words down J-Lens's list instead of 10; ran the template lens on each part of the split (CW-13, 50 and 25 prompts, no GPU) | Bridge in J-Lens's top 200 for 35 of 50, against about two thirds of single write-ups (oracle 133/200, NLA 127/200). Template lens finds the bridge in its top 50 from the rest in 16 of 25, from the J part in 15 | Row 1's gap is largely the top-10 cutoff. The bridge is also in the rest, and the readers pick it up there little, the NLA not more than the oracle lens (12% against 20%, difference within noise) | this repo | Fair: asked for after CW-12, one lens |
| 9 | Re-scored chain at L56 with WorkspaceBench's rule: the number given as the computed result (his check C1, 60 items) | Oracle 11, J-Lens 11, NLA 1; decoy null 0.02–0.04. Under the any-mention rule it was 36 / 25 / 13 | The oracle lens reads the number, at a third of the any-mention size and no more often than J-Lens | Vishesh, on this repo's readouts | Fair: one readout per cell |
| 10 | Repeated row 7's split on the 70 multihop prompts it had not used, with the cutoff, scoring and three tests fixed before the run (CW-14, layer 42, 2 write-ups per prompt) | Names the step from whole / J part / rest / random part: oracle lens 86 / 93 / 21 / 44%, NLA 85 / 88 / 23 / 49%. J part minus rest: oracle +0.71 [0.61, 0.81], NLA +0.65 [0.55, 0.74]; oracle J part minus random part +0.49 [0.37, 0.59]; each p < 0.0001, also when prompts are grouped into their 26 families | Row 7 holds on new prompts for both readers. The NLA does not differ from the oracle lens on any part, so row 7's "NLA does better on a random part" did not repeat | this repo | Firm for multihop at layer 42 (off the oracle lens's trained layers); the J part is by construction the part that maps to output words |
| 11 | Vishesh's rerun of the split (his 007, pre-registered; brew dev and 49 held-out test tables, multihop items 51–100, oracle lens at its trained layers 40 and 44, likelihood scoring; NLA rerun post hoc) | Brew: the oracle lens's tracking of the hidden colour is carried by the J part at L40, confirmed on test (J 5.0 nats against whole 2.9); L44 unresolved. The NLA's is not concentrated in J (J 0.95 against whole 1.86; removing J costs no more than removing a random slice; post hoc, dev only). Multihop naming, oracle L40 / L44: J part 91% / 94%, rest 42.5% / 12.5%. Model side: swapping the J part between paired brew prompts does about what a matched random part does at L42 (mixed at L40, L44). Chain, 7 items, L56: the oracle lens's number signal is outside J. Mean removal lowers oracle naming on multihop (85% to 45%) | Row 10 is repeated for the oracle lens by separate code at trained layers. J dependence differs by reader and task: on brew the NLA does not rely on the J part. No sign that the model gives the J part extra causal weight | Vishesh | His report `docs/reports/007_jsubspace.md` and `results/007_jsubspace/status/` on branch `007-jsubspace`; read 2026-10-06 |
| – | Reader-free label for "the model uses this content here" | Reliable for whole positions; not for specific content (two fits disagree) | The yardstick the project wanted does not exist yet | Vishesh | Firm as a negative |

His "rest" (row 4) and ours (row 7) are different splits: his is everything outside 16 directions per concept, about 95% of the concept; ours is everything outside J-Lens's top 1,024 directions. Much of his "rest" may lie inside our J part. Not checked.

## Current conclusion

**Neel's question.** No evidence that the NLA reads content outside the J part that the oracle lens does not. On natural prompts both readers get the bridge from the J part (CW-12); the bridge is also present in the rest (template lens, CW-13) and the NLA names it from there in 12% of samples against the oracle lens's 20% (NLA − oracle −0.08 [−0.20, +0.03], 30 items: not distinguishable). On planted concepts (the collaborator, 45 test concepts) the oracle lens and a linear probe read the non-J part better than the NLA. The NLA does name the bridge more often than the oracle lens on damaged activations: a random 1,024-dimensional part alone (35% against 17%, +0.18 [+0.08, +0.32], 30 items) and after the bridge's J-Lens directions are deleted (53% against 37%, +0.16 [+0.06, +0.27], 50 items, CW-11). Why is not tested. These paired differences were computed on 2026-10-05 after the results were known (`scripts/recount_nla_vs_oracle.py`, `runs/recount/analysis.txt`; the same script recounts CW-1, CW-11 and CW-12 from the raw files with separately written matching code and reproduces them). Limits: 30 prompts for the split, one NLA checkpoint, two different definitions of the J part. CW-14 (70 new prompts, tests fixed in advance) confirms the J-part result for both readers and finds no NLA-minus-oracle difference on any part (random part +0.05 [−0.04, +0.13]), so the earlier "NLA names the bridge more often on a random part" is withdrawn. the collaborator's 007 stage 1a (brew, 48 dev tables, oracle lens at its trained layers 40 and 44, likelihood scoring, 10 random draws) finds the same for the oracle lens. What this does not show: that the J part is a workspace the model uses. The J part is the subspace that maps to output words, and removing it at L42 changes the model's own answer no more than removing a random part (CW-14 add-on, rough).

Whether the prose readers re-solve is open (below, by task).

- **multihop (one-line trivia):** they name the hidden step about twice as often as J-Lens's top 10 at L42 (largely the top-10 cutoff: CW-13), and they also name it at L28, where J-Lens shows it in 1/50 (the NLA is off its training layer at L28). Deleting the step's J-Lens directions lowers naming (oracle lens 65% → 37% of samples, NLA 65% → 53%), so both read those directions in part, the oracle lens more. What survives deletion is **not** good evidence of re-solving: CW-11 shows that deleting the J-Lens directions of the prompt's own words barely stops the readers reporting those words (CW-11 below). "Mostly re-solving" is withdrawn; how much is re-solved is open.
- **chain (arithmetic), at L56 where J-Lens shows the number:** the oracle lens names the number in 36/60 (chance 12), in 31 of those without restating the starting number. Reading. The NLA is near chance. Under WorkspaceBench's stricter rule (the collaborator's check C1, his `docs/reports/checks_2026-10-04_no_gpu.md`) it is oracle 11/60, J-Lens 11/60, NLA 1/60, null 0.02–0.04: still reading, a third of the size, and not more than J-Lens.
- **brew (10-rule colour table), at ":" L42:** both readers rarely state the hidden colour in its role (2–5 of 50), and those mentions go to about zero when its directions are deleted. No sign of re-solving.
- **NLA text:** about 60% of its concepts on multihop are prompt wording or format; 66/98 readouts contain at least one claim the prompt rules out. The NLA rarely names a hidden step the oracle lens misses: 3 of 100 multihop prompts at L42 against 8 the other way (one readout each), 3 of 50 against 6 with 4 samples, 0 of 30 against 1 with 2 samples. On intact multihop and chain activations its total is at or below the oracle lens's; on brew it is mixed (ours 5 against 2 or 3 against 5 by judge; the collaborator's 5–15% against 0–3%).

The explanation "a reader re-solves when it can rebuild the prompt" has one weak support (CW-10: NLA readouts that recover more of the prompt name the step more often) and no causal test: CW-11's cue removal did not hide the prompt, so it could not test it.

the collaborator's later brew results (his repo STATUS, 2026-10-05; not in the shared doc's log yet): scored by likelihood, the NLA favours z at ":" (+1.86 nats [1.41, 2.34], 44/48 tables), but the model's own output at that cell already favours z (runner-up in 56% of tables), so no held intermediate beyond the output is shown.

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
  - **Trained layers (added 2026-10-05).** The oracle lens's README (`agu18dec/olens_and_ar`, read 2026-10-05) gives its trained layers as 20–60 in steps of 4 and says "Layers outside 20–60 step 4 are off-contract. The student never saw them." The NLA was trained at L42 only. So no layer used here is a trained layer for both readers:

    | layer | oracle lens | NLA |
    |---|---|---|
    | 16 | off | off |
    | 28, 36, 44, 56 | on | off |
    | 42 | off | on |

    Every L42 result below (main comparison, deletion, CW-11, CW-12, brew) has the oracle lens off its trained layers. The one check so far: on the 100 multihop items the oracle lens names the bridge in 80 at L42 and 72 at L44 (one readout each, same word match; named at both 68, at neither 16, only L42 12, only L44 4; `runs/multihop100/olens_L42,44.multihop.jsonl`). Checked 2026-10-05: the WorkspaceBench `OLens` class used for every oracle-lens call here has the README's prompt text (`concepts_raw`, word for word), transform (`unit`) and alpha (16000); the checkpoint's `run.json` lists the same. So the only departure from its contract is the layer.
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
|   | Same 100 items, oracle lens at a trained layer | 100 multihop | 44 | — | 72 | — | [multihop100](runs/multihop100/) |
| 2 | Earlier layers | 50 multihop | 16 / 28 / 36 | 0 / 1 / 9 | 9 / 26 / 33 | 11 / 27 / 28 | [sweep50](runs/sweep50/) |
| 3 | Delete the step's J-Lens directions (2–6 per item); random-direction control unchanged | 50 multihop | 42 | 15 → 0 | 30 → 17 | 36 → 25 | [abl50](runs/abl50/) |
| 4 | Delete the step's neighbourhood (cos > 0.8, ~31 directions). Control = same number of random directions | 50 multihop | 42 | 15 → 0 | 33 → 20 | 34 → 23 | [abl3](runs/abl3/) |
| 5 | Swap: delete own step, add another item's step | 50 multihop | 42 | planted 14 | own 20, planted 7 | own 20, planted 6 | [abl3](runs/abl3/) |
| 6 | chain at the NLA's training layer (oracle lens off its trained layers) | 60 chain | 42 | 6 (chance 4) | 27 (chance 21) | 9 (chance 12) | [chain60](runs/chain60/) |
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
- **Not ruled out.** The NLA is off its training layer at L56; the oracle lens is on a trained layer there. Readouts contain many numbers (oracle 6–9, NLA 4–5 distinct per readout), hence the high chance rates. Predictions were written before the L42 run; the L56 run was added after seeing that J-Lens peaks there.

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
|---|---|---|---|---|---|
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

Design, committed before the run: `designs/CW-12_j_split.md`. Code: `jobs/cw12_a_sweep.py`, `jobs/cw12_b_split.py`, `jobs/cw12_c_nla.py`, `jobs/cw12_d_pca.py`, `jobs/cw12_e_nla.py`, `scripts/worker.py`, `scripts/cw12_analyze.py`, `scripts/cw12_probe.py`, `scripts/cw12_fig.py`. Raw: `runs/cw12/`. Figure: `figs/cw12_split_final_30items.png`.

**Setup.** Qwen3.6-27B, L42, the first 30 of the 50 multihop items of CW-4, same saved activations (bf16 capture). J part = projection onto the top 1,024 principal directions of all unit-normalised J-Lens token directions (W_U[t] J) at L42; these hold 93% of the spread of those directions. The basis is in the public repo (`jbasis/`). k = 1,024 was chosen from a 5-item check (`runs/cw12/sweep.jsonl`: at k = 64 and 256 J-Lens still reads its tokens from the rest; at 1,024 it does not). Controls: a random 1,024-dimensional subspace (seed 0), and, added after the first results, the top 1,024 PCA directions of 16,000 L42 activations from benchmark prompts (all token positions; multihop, association, basic readout, multilingual, chain, brew), centred. Readers: oracle lens and NLA (`nla-rl`, `iter_000400`), 2 samples per cell, fixed seeds; both rescale their input. Word match.

**Measured** (share of the 60 samples per cell that name the bridge, 95% bootstrap over items; prompt recovery as CW-10):

| part given to the reader | share of squared norm | oracle: bridge | NLA: bridge | oracle: recovery | NLA: recovery |
|---|---|---|---|---|---|
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
- On natural prompts there is no sign that the NLA reads the bridge from outside the part J-Lens reads. This agrees with the collaborator's planted-concept result (summary row 4).
- **Most likely objection:** a part of an activation is off-distribution for a reader trained on whole activations, so "the reader gets little from the rest" may be a failure to parse it and not absence of use. The random control argues against this (its complement, also a part, reads fine), but the rest has a different statistical character from a random complement. Not further addressed.
- n: 30 items, 2 samples per cell, one prompt family, one layer, one definition of the J part, word match (hand-checked on other readouts). The probe labels are coarse; no probe for the bridge itself (one item per bridge).

### CW-13: the bridge's rank in J-Lens's full list, and the template lens (2026-10-05, no GPU)

Code: `scripts/cw13_jrank.py`, `scripts/cw13_template.py`, `scripts/cw13_fig.py`. Output: `runs/jrank/`. Figures: `figs/cw13_bridge_rank.png`, `figs/cw13_where_is_the_bridge.png`. Not pre-registered; both analyses were chosen after CW-12's results were known.

**Setup.** The 50 multihop items with saved L42 activations (CW-4), and the 30 split into parts in CW-12. J-Lens scores for all 248,320 tokens, computed on CPU from the neuronpedia map and the model's output-word matrix (same cosine readout as the benchmark). The bridge's rank is the best rank among tokens whose text equals a bank intermediate (letters and digits, case-insensitive); 46 of 50 items have such a token. Template lens: `camilablank/workspace-lenses` @ d740106d, 13,174 words, layer 42, scored as its README states, by cosine of the activation against each word's template direction. The bridge is in its vocabulary for 41 of 50 items (25 of the 30).

**Measured.**
- J-Lens, 50 items: bridge within the top 10 in 15 (30%), top 50 in 24 (48%), top 200 in 35 (70%), top 1,000 in 38. Median rank 47.
- Template lens, 50 items: top 10 in 14, top 50 in 31, top 200 in 35. Median rank 20.
- For comparison, on the same 50 items the oracle lens names the bridge in 66% of single readouts and the NLA in 64% (CW-11, intact, 4 samples).
- By part (30 items), bridge within the top 50: J-Lens 10 whole, 10 J part, 0 the rest (median rank in the rest 77,789). Template lens (25 items) 18 whole, 15 J part, 16 the rest (median ranks 28, 26, 37); random 1,024 directions 8, all but those 16; PCA top 1,024 18, all but those 5.

**Taken to mean.**
- "The prose readers name the bridge about twice as often as J-Lens" (results 1 and CW-1) is largely the top-10 cutoff. Looking 200 words down J-Lens's list gives the bridge as often as one prose readout does.
- The bridge is still present in the rest of the activation: the template lens finds it there as often as in the J part. J-Lens finding nothing in the rest is by construction. So in CW-12 the prose readers get little of the bridge from the rest although it is there to be read.
- For Neel's question: there is bridge content outside the J part, and the NLA does not pick it up better than the oracle lens (12% against 20%).
- **Most likely objection:** 25 to 30 items, and one lens. A first pass used an unnormalised projection in place of the lens's stated cosine rule; the numbers here are from the stated rule (the earlier ones were 14, 12 and 11 of 25, the same pattern).

### CW-14: the split repeated on 70 new prompts with tests fixed in advance (2026-10-05)

Design, with its amendments in order: [designs/CW-14_split_confirmation.md](designs/CW-14_split_confirmation.md) (committed at `1791322` before any data). Raw rows: [runs/cw14/](runs/cw14/). Analysis: [analysis.txt](runs/cw14/analysis.txt), [analysis_family.txt](runs/cw14/analysis_family.txt). Plot: [figs/cw14_split_70_new_prompts.png](figs/cw14_split_70_new_prompts.png).

- **Setup.** Qwen3.6-27B bf16, layer 42, read position of the bank. The 70 multihop items not in CW-12 (items 31–100; 26 item families, the largest has 15 prompts). J part = projection on the top 1,024 directions of the one CW-12 J basis; one random 1,024-dimensional subspace (seed 0). Oracle lens and NLA, 2 samples per cell, word match. Fresh captures reproduce the saved activations on the 50 overlapping items (cosine 1.00000).
- **Measured** (share of samples naming the hidden step, 95% bootstrap over prompts):

  | part | share of squared norm | oracle lens | NLA |
  |---|---|---|---|
  | whole | 1.00 | 0.86 [0.79, 0.93] | 0.85 [0.76, 0.92] |
  | J part | 0.16 | 0.93 [0.86, 0.98] | 0.88 [0.81, 0.94] |
  | rest | 0.84 | 0.21 [0.14, 0.30] | 0.23 [0.14, 0.32] |
  | random part | 0.21 | 0.44 [0.34, 0.56] | 0.49 [0.38, 0.60] |

- **Pre-set tests** (paired over 70 prompts, sign-flip permutation; rule: confirmed at p < 0.005 with CW-12's sign):
  1. Oracle lens, J part − rest: +0.71 [0.61, 0.81], p < 0.0001, prompts +/0/−: 57/12/1. Confirmed.
  2. NLA, J part − rest: +0.65 [0.55, 0.74], p < 0.0001, 54/16/0. Confirmed.
  3. Oracle lens, J part − random part: +0.49 [0.37, 0.59], p < 0.0001, 38/32/0. Confirmed.
  - Grouped by the 26 item families (bootstrap and sign-flip over families; chosen after the run, because prompts in a family share a template): +0.71 [0.62, 0.83], +0.65 [0.55, 0.73], +0.49 [0.34, 0.63], each p < 0.0001.
- **Secondary.** NLA J part − random part +0.39 [0.28, 0.49] (CW-12: +0.07, not shown). Random part − rest: oracle +0.23 [0.12, 0.34], NLA +0.26 [0.16, 0.37]; not predicted. J part − whole: oracle +0.06 [0.00, 0.13], NLA +0.03 [−0.03, 0.09]. NLA − oracle by part (by family): whole −0.01 [−0.12, 0.06], J part −0.05 [−0.12, 0.01], rest +0.01 [−0.05, 0.12], random part +0.05 [−0.04, 0.13]. On the whole activation the step is named by both for 61 prompts, oracle only 5, NLA only 1, neither 3. J-Lens top-10 names it for 36 (whole), 37 (J part), 0 (rest), 20 (random part) of 70.
- **Differences from CW-12.** The effect is larger (+0.71 against +0.28) and whole-activation naming is higher (0.86 against 0.45): these 70 prompts are other item types and easier for the readers. CW-12's NLA-over-oracle gap on the random part (0.35 against 0.17) is not there.
- **Is the rest read badly because it is unusual input?** Checked after the run: the rest has cosine 0.92 with the whole activation and holds 95% of the mean activation; the J part has cosine 0.40 and holds 5% of the mean. The most ordinary-looking input is read worst.
- **Add-on, model side (exploratory, rough;** [modeluse.jsonl](runs/cw14/modeluse.jsonl)**).** Replacing the L42 read-position activation by one part and reading the model's next token, 63 of the 70 prompts whose unpatched top token starts the target: removing the J part changes the log-probability of that token by −0.03 [−0.12, +0.04]; removing random parts −0.09 and −0.13; removing the unembedding part −0.12; keeping only the 10% outside the PCA part −3.65. One layer and one position, so the model can route around it. It gives no sign that the model needs the J part there.
- **Unembedding part.** The top 1,024 directions of the unit unembedding rows overlap the J part by 0.32 (0.20 is unrelated), so the J part is not the plain output-embedding subspace ([ubasis_overlap.txt](runs/cw14/ubasis_overlap.txt)).
- **Most likely objection.** The J directions are built from the Jacobian to the output, so the J part is the part of the activation that maps to words. "Readers that write words read the word-mapped part" may be all this shows; it does not make the J part a workspace. Not addressed here. the collaborator's 007 stages 2 and 2b (swaps between paired brew prompts) test the model side.
- **Extension (secondary; four more conditions on the same 70 prompts, added before any reader output existed).**

  | part | share of squared norm | oracle lens | NLA |
  |---|---|---|---|
  | all but the random part | 0.79 | 0.84 [0.76, 0.91] | 0.84 [0.76, 0.91] |
  | second random part (seed 1) | 0.19 | 0.55 [0.44, 0.66] | 0.54 [0.44, 0.65] |
  | PCA part | 0.90 | 0.84 [0.76, 0.91] | 0.79 [0.70, 0.87] |
  | all but the PCA part | 0.10 | 0.06 [0.01, 0.11] | 0.16 [0.09, 0.24] |

  - Removing the J part costs far more than removing a random part of the same size: rest − all-but-random −0.62 [−0.73, −0.51] (oracle), −0.61 [−0.70, −0.51] (NLA), each p < 0.0001 (CW-12: −0.25 and −0.27).
  - The random-part result does not depend on the draw: J part − second random part +0.38 [0.28, 0.48] (oracle), +0.34 [0.24, 0.44] (NLA); the two draws differ by −0.11 [−0.21, 0.00] and −0.05 [−0.15, 0.05].
  - J part − PCA part +0.09 for both readers (oracle [0.03, 0.15], p = 0.014; NLA [0.02, 0.16], p = 0.020): weak. The PCA part holds 90% of the squared norm and overlaps the J part by 0.31, so it contains much of it.
  - Not run: a centred-activation condition (prepared at the collaborator's suggestion, held back).

### CW-15: what the readers write from each part (2026-10-06, no GPU, exploratory)

Asked for by the project's TA (2026-10-06): a systematic look at the content of readouts, and whether what the NLA gets from outside the J part is useful or hallucination. Uses CW-14's saved readouts (70 prompts, 2 samples, L42). Code: `scripts/cw15_readout_content.py`, `scripts/cw15_fig.py`. Output: [runs/cw15/content.txt](runs/cw15/content.txt). Plot: [figs/cw15_content_by_part.png](figs/cw15_content_by_part.png). Random examples to read: [to_read/readouts_by_part_8_random.md](to_read/readouts_by_part_8_random.md) (8 prompts, seed 0). The measures were chosen after reading 3 random prompts' readouts.

- **Read first (3 random prompts, then 8).** From the J part both readers keep the entities (Sahara, Africa; Italy, Rome) and invent the framing (a flashcard, a forum, a multiple-choice question). From the rest they keep the framing (a trivia sentence, "The capital of the … is") and write other entities (the Great Wall of China, Tokyo, Philadelphia). From a random part some topic words survive inside unrelated framing (malware, an email).
- **Measured** (mean over 70 prompts [95% bootstrap]; word-level, no judge):

  | | oracle: J part | oracle: rest | NLA: J part | NLA: rest |
  |---|---|---|---|---|
  | names the hidden step | 0.93 | 0.21 | 0.88 | 0.23 |
  | names the answer | 0.91 [0.84, 0.97] | 0.26 [0.18, 0.36] | 0.89 [0.81, 0.96] | 0.09 [0.04, 0.14] |
  | share of the prompt's content words repeated | 0.47 [0.43, 0.51] | 0.21 [0.17, 0.26] | 0.45 [0.42, 0.49] | 0.29 [0.25, 0.32] |
  | NLA quotes a final word and it is the prompt's last word | – | – | 0.83 [0.75, 0.91] | 0.98 [0.96, 1.00] |

  The NLA quotes a final word or phrase in 81–83% of readouts from the whole activation, the J part and the rest. The prompts end in "is" (49), "the" (20) or "of" (1); scored against another prompt's last word the rate is 0.50–0.60. From the rest it is right for 97% of "the" prompts and 99% of "is" prompts; from the J part 65% and 93%; from a random part 52% and 73%; from the 10% outside the PCA part 1%.
- **Taken to mean.** What the NLA reads from outside the J part is real but it is the form of the sentence (the current word, the trivia frame), not the content; the entities it writes from there are mostly wrong. The J part carries the content words and less of the current word. This fits the J part being built from directions that map to later output words.
- **Not ruled out / limits.** One measure of "form" (the last word) and it exists only for the NLA, whose training makes it describe the prompt; the oracle lens has no comparable habit. A count of made-up names by capitalised words gave inconsistent results across readers and is not used. No judge of which statements are false. Three function words is a small set to tell apart. Multihop only; the collaborator finds the NLA does not rely on the J part on brew.

**CW-15, part 2: what a write-up is made of, on whole activations** (`scripts/cw15_whole_content.py`, [runs/cw15/whole_content.txt](runs/cw15/whole_content.txt)). The 100 multihop prompts of the main comparison, 1 write-up each. Every content word goes in one bucket by word match (first match wins): hidden step, answer, prompt word, format word (a fixed list: quiz, sentence, token, final, …), other.

| | content words per write-up | hidden step | answer | prompt words | format words | other |
|---|---|---|---|---|---|---|
| NLA, L42 | 80 | 0.03 | 0.02 | 0.17 [0.16, 0.18] | 0.41 [0.40, 0.42] | 0.37 |
| Oracle lens, L42 | 35 | 0.07 | 0.08 | 0.34 [0.31, 0.37] | 0.04 [0.03, 0.05] | 0.47 |
| Oracle lens, L44 (trained layer) | 34 | 0.07 | 0.09 | 0.34 [0.31, 0.36] | 0.04 [0.04, 0.05] | 0.46 |

- In words per write-up: hidden step and answer about 4 (NLA) and 5 (oracle lens); prompt words about 14 and 12; format words about 33 and 1.5.
- The commonest "other" words are more format for the NLA (sets up, puzzle, clue, logic, immediate) and topic words for the oracle lens (city, located, capital, currency, national).
- The oracle lens writes the same mix at L42 and L44.
- **Taken to mean.** On whole activations the two readers carry about the same amount of task content. What the NLA adds is a description of the prompt's form. With the split result above (form is what it reads from outside the J part, correctly), this is the one thing found so far that the NLA reports and the oracle lens does not: the form of the prompt. Exploratory; the format list was written for the NLA's style in the pilot, and "other" is not judged for truth.

### CW-16: what the NLA's text rebuilds, through its reconstructor (2026-10-06, exploratory)

Suggested by the project's TA. Design: [designs/CW-16_reconstructor.md](designs/CW-16_reconstructor.md) (committed before the run). Texts: [runs/cw16/texts.jsonl](runs/cw16/texts.jsonl). Output: [analysis.txt](runs/cw16/analysis.txt), [analysis_ctrl.txt](runs/cw16/analysis_ctrl.txt). Code: `scripts/cw16_*`.

- **Setup.** The NLA's reconstructor (`ceselder/qwen3.6-27b-nla-rl`, `ar_reconstructor/`, end-of-run checkpoint) maps text to an L42 activation; forward pass re-implemented from EasyNLA's `critic_predict`. Texts are CW-14's NLA write-ups for the 70 multihop prompts (2 samples) and edited versions of them; targets are the saved activations. Vectors scaled to norm sqrt(5120) as in its loss. FVE is against the mean of the 70 targets (a target's cosine with that mean is 0.85), so it is a harder baseline than the 77% on web text in the NLA's README. LLM edits: one `claude -p --model sonnet` call per write-up, 140 of 140 parsed, not validated.
- **Loader checked against the official one (2026-10-07).** The same 3,376 texts through EasyNLA's own `NLACriticModel.from_pretrained` and `critic_predict` (github.com/asherps/EasyNLA @ `4d72847`) and through `scripts/cw16_ar.py` on one pod: cosine between the two outputs at least 0.999999 for every text, and every FVE below is the same to four decimals ([runs/cw16c/compare.txt](runs/cw16c/compare.txt)).
- **Gate passed.** Original write-up: cosine 0.93, FVE +0.41 [0.36, 0.46]. Another prompt's write-up: cosine 0.74, FVE −1.05.
- **Measured** (mean over 70 prompts):

  | text | FVE | FVE in J part | FVE in rest |
  |---|---|---|---|
  | original write-up | +0.41 | +0.59 | +0.31 |
  | the prompt itself | +0.19 | +0.41 | +0.06 |
  | false statements removed (LLM) | −0.03 | +0.28 | −0.20 |
  | random sentences deleted to that length (post hoc control) | −0.10 | +0.15 | −0.23 |
  | content only (LLM) | −0.33 | +0.03 | −0.53 |
  | random sentences deleted to that length (post hoc control) | −0.35 | −0.09 | −0.50 |
  | form only (LLM) | −0.90 | −0.59 | −1.08 |
  | hidden step swapped for another prompt's (62 prompts) | +0.34 | +0.52 | +0.24 |
  | oracle lens write-up for the same activation | −0.29 | +0.15 | −0.54 |

- **The TA's question, first attempt (superseded).** An LLM was told to delete "false statements", and random sentences were deleted as a control. Both lowered FVE by about the same amount. Reading the examples showed two faults: the instruction lumped wrong facts together with the NLA's inexact restatements of the prompt, and the control deleted too much. Redone as CW-16b below; the rows "false statements removed", "content only", "form only" and the two controls in the table above are from this first attempt.
- **CW-16b (2026-10-06): labelled spans, length-matched controls.** After reading 7 more write-ups, an LLM (`claude -p --model sonnet`) cut each write-up into spans and labelled each as format, restatement of the prompt (right or wrong in meaning), claim about the topic or answer (right or wrong), or other; spans must reproduce the write-up exactly (138 of 140 write-ups passed; not validated by hand beyond reading 3). Versions are built from the labels by code. Controls remove random spans of the same number of characters (within 1–2 points), 3 draws each. A 10-prompt pilot was read before the full run. Code: `scripts/cw16b_*`; output [runs/cw16b/analysis.txt](runs/cw16b/analysis.txt); plot [figs/cw16b_what_the_rebuild_needs.png](figs/cw16b_what_the_rebuild_needs.png).
  - Make-up of a write-up by characters (first 20 write-ups): correct claims 29%, format 26%, wrong restatements 24%, correct restatements 11%, wrong claims 9%.
  - FVE of the original write-up +0.41 (J part +0.60, rest +0.31), as before.

  | what is removed | FVE after removal | FVE of length-matched random removal | difference [95%] |
  |---|---|---|---|
  | wrong claims (56 prompts have any) | +0.38 | +0.40 | −0.02 [−0.03, +0.00] |
  | wrong restatements of the prompt | +0.16 | +0.34 | −0.18 [−0.24, −0.13] |
  | all restatements of the prompt | −0.25 | +0.26 | −0.51 [−0.59, −0.43] |
  | all claims about topic and answer | +0.16 | +0.27 | −0.11 [−0.16, −0.07] |
  | format statements | +0.40 | +0.31 | +0.10 [+0.06, +0.15] |

  - **Wrong parts corrected in place** (same length): FVE +0.47 against +0.41, difference +0.05 [+0.03, +0.08]; in the J part +0.02 [−0.00, +0.05], in the rest +0.07 [+0.04, +0.10].
  - **Taken to mean.** For this reconstructor on these prompts: (1) correcting the false statements gives a small gain, mostly outside the J part; it does not shift the rebuild toward the J part. (2) Wrong claims about the topic matter no more than an average piece of text. (3) The restatement of the prompt is the part the rebuild depends on most, and its inexact versions still carry more than an average piece of text, most of it outside the J part. (4) Format statements carry nothing here, which fits all 70 prompts sharing one format.
  - **Not ruled out.** LLM labels are unvalidated and the wrong/right line for restatements is a judgement. Removing spans leaves broken sentences, which the controls share. One reconstructor, one prompt family.
- **Other observations.** The write-up rebuilds more than the prompt text does (+0.41 against +0.19), though the reconstructor was trained on write-ups, not prompts. Swapping the hidden-step word changes little (−0.06) and does not move the J part toward the other prompt (centred cosine with the partner's target −0.04 → −0.02). Write-ups the NLA made from the J part alone rebuild the J part of the real activation almost as well as whole-activation write-ups (centred cosine 0.72 against 0.79); from the rest 0.40; from a random part 0.36.
- **Prediction that failed.** Form-only text was expected to rebuild the rest; it rebuilds nothing beyond the mean. All 70 prompts share one form, so form may sit in the mean that FVE subtracts. CW-15's "the rest holds the form" is not confirmed from this direction.
- **Most likely objection.** Every edited text is off the reconstructor's training distribution (shorter, different style), and the oracle lens write-up is another style again; lower FVE for those may be style, not content.

### CW-17: poetry and directed modulation, stopped at the readability check (2026-10-06)

Design: [designs/CW-17_poetry_split.md](designs/CW-17_poetry_split.md) (committed before the run). Asked for by the collaborator. Outputs: [runs/cw17/](runs/cw17/).

- **Planned.** The J part / rest split on WorkspaceBench poetry (100 items; content = the planned rhyme word, read at the newline ending line one), NLA at L42 and oracle lens at L44 (its trained layer, with a J basis built for L44: top-1,024 share 0.91, overlap with the L42 basis 0.90; the L42 basis rebuilt on the pod matches the saved one, overlap 1.0000).
- **Stopped early.** The first write-ups (6–7 prompts per reader, read by hand) do not name the rhyme word: oracle lens 0 of 7 prompts from the whole activation, NLA 1 sample of 12. J-Lens top-10 has the rhyme word for 0 of 100 prompts at L42 and at L44, in every part. With nothing readable there is nothing to split, so the reader run was stopped and the pre-set tests were not run. Partial reader outputs are kept (`olens_poetry_a.jsonl` 32 rows, `nla_poetry_*.jsonl`).
- **Where the rhyme word is readable** (J-Lens, 100 prompts, [poetry_layers.txt](runs/cw17/poetry_layers.txt)): at the bank's read position it is in the top 10 for at most 2% of prompts at any layer from 20 to 62 (median rank 6,652 at L42, about 1,000 at L60). At the last prompt token it is in the top 10 for 3% at L42, 13% at L56, 62% at L60, 96% at L62, i.e. only where it is about to be output.
- **Directed modulation layer check** (J-Lens rank of the held word at the writing positions, [dm_layers.txt](runs/cw17/dm_layers.txt)): only 19 of 100 items have a single-token concept, which is all this method scores (9 "think", 9 "do not think", 1 "suppress"). "Think": in the top 10 at some writing position for 22% of prompts at L40–44 and 56% at L60; median best rank 4,582 at L42, 14 at L56, 7 at L60. "Do not think": 0% up to L56. This agrees with the collaborator's earlier study (readable mainly at L51–59).
- **Correction (2026-10-07): the poetry conclusion is not reliable.** This run did not follow the benchmark's way of reading poetry: it read the oracle lens at one layer (44) where the benchmark reads every trained layer and passes an item if any layer names the word; it scored by exact word match where the benchmark uses its judge; and "not readable" rests on J-Lens ranks from our own script. The benchmark's notes for this family say that at this position "the O-lens already writes line two with" the committed rhyme. So what is established is only that J-Lens ranks the rhyme word low there and that 6–7 prompts at L44 did not contain the exact word. Poetry was rerun through the benchmark's own producer and judge: see CW-19. The directed-modulation check scores 19 of 100 items and is a small agreement with the collaborator's earlier study, not a result of its own.
- **Observation, 6 prompts, not a result:** the NLA names the last word of line one in 42% of samples from the whole activation, 10% from the J part and 0% from the rest.
- **Limits.** "Readable" here means J-Lens top 10; a reader could still name a word J-Lens ranks low (as on multihop, where the readers beat J-Lens's top 10). The early stop rests on 6–7 prompts per reader plus the J-Lens count on all 100.

### CW-18: which families' content is readable around layers 40–44 (2026-10-07, no reader)

Done to choose where a two-reader split can run (NLA at L42, oracle lens at L40/44). One base-model pass over 100 items of each family at the bank's read position, layers 20–62; J-Lens (neuronpedia) rank of the bank's `intermediates` word, single-token words only. `jobs18/cw18_layers.py`, `scripts/cw18_ranks.py`, [runs/cw18/ranks.txt](runs/cw18/ranks.txt).

Median rank / share in top 10 / share in top 200:

| family (items scored) | L36 | L42 | L44 | L52 | L60 |
|---|---|---|---|---|---|
| multihop (91), reference | 82 / 0.27 / 0.60 | 19 / 0.44 / 0.86 | 21 / 0.43 / 0.84 | 14 / 0.43 / 0.88 | 8 / 0.56 / 0.96 |
| association (79) | 349 / 0.14 / 0.41 | 50 / 0.35 / 0.67 | 63 / 0.34 / 0.66 | 336 / 0.10 / 0.42 | 150 / 0.19 / 0.56 |
| typo (100) | 205 / 0.15 / 0.48 | 10 / 0.50 / 0.85 | 30 / 0.38 / 0.76 | 240 / 0.05 / 0.46 | 212 / 0.04 / 0.48 |
| multilingual (100) | 228 / 0.23 / 0.49 | 15 / 0.45 / 0.85 | 18 / 0.45 / 0.82 | 5 / 0.58 / 0.95 | 1 / 0.92 / 0.99 |
| basic readout (88) | 81 / 0.27 / 0.58 | 9 / 0.52 / 0.73 | 7 / 0.52 / 0.72 | 14 / 0.35 / 0.92 | 1 / 0.94 / 1.00 |

- All four are readable at L42–44 about as well as multihop (poetry: median rank 6,652 at L42; directed modulation: 4,582). Typo and association peak at L40–44 and fall away by L52, so their content is not just the next output word; multilingual and basic readout keep rising toward the output.
- 21 association and 12 basic-readout items are not scored (multi-token word, or 7 basic-readout ids not found in the bank file).
- Not a reader result. It says where a split with both readers is worth running.


### CW-19: poetry read the benchmark's way (2026-10-07, exploratory)

Qwen3.6-27B, WorkspaceBench `poetry`, all 100 prompts, read at the newline that ends line one; the target is the rhyme word the model goes on to use at the end of line two. Oracle lens through the benchmark's own producer (`p.run_family`) at layers 40 and 44 (two of its trained layers, not all of them); NLA at layer 42, 1 sample per prompt. Scored by the benchmark's judge (`wsbench judge`, `google/gemini-3.8-flash`, pinned bank). Jobs: [jobs19/](jobs19/). Outputs and judge cells: [runs/cw19/](runs/cw19/). Counts: [runs/cw19/counts.txt](runs/cw19/counts.txt) from `scripts/cw19_count.py`.

- **Measured.** Judge passes out of 100 prompts: oracle lens 4 at L40, 11 at L44, 15 at either; NLA 30 at L42. Both 6, NLA only 24, oracle only 9.
- **Where the NLA's passes come from.** Every NLA write-up quotes a version of line one. In 43 of 100 a quoted line ends in the real last word; in 23 of 100 a quoted line ends in the target rhyme word instead (e.g. real line ends "wheat", write-up quotes a line ending "heat"). 23 of the 30 passes are this case; the others are mostly the target appearing in a list of example rhymes. Seen in 8 randomly drawn passes (6 of 8 in a quoted line, 2 of 8 in a rhyme list).
- **Oracle passes read by hand (all 15).** The target usually appears somewhere inside a written line two, not as its rhyme word, and several passes are a different form of the word ("lead" for "led", "shone" for "shine", "died" for "dead").
- **Control with other rhyme words (added after seeing the scores, so exploratory).** For each prompt, 3 other words that rhyme with line one's last word (CMU dictionary via `pronouncing`), closest in word frequency to the target (`scripts/cw19_alt_rhymes.py`, which writes the altered banks; 3 prompts have fewer than 3; mean Zipf frequency 5.08 target, 4.93 controls). Same judge, same readouts, target swapped for a control word ([runs/cw19/ctrl/](runs/cw19/ctrl/), counts in `ctrl/counts.txt`). NLA at L42: target 30 of 100, the three control sets 6, 8 and 10. Paired difference per prompt, target minus mean control: +0.22 [0.14, 0.30] (bootstrap over 100 prompts). Oracle lens at L40 or L44: target 15, controls 7, 3 and 10; difference +0.08 [0.01, 0.16].
- **The benchmark's own floor.** `evals/baselines/prompt_only.json` gives 0.71 [0.62, 0.80] for poetry: the stock model given the prompt text and no activation, same judge. It was measured at the family's older read position (the final prompt token) and the benchmark says it is being re-measured for the current one. Both readers are far below it.
- **What this does and does not show.** The readers name the model's rhyme word more often than another rhyme word of similar frequency, so the passes are not explained by writing arbitrary rhymes. It does not show they read a planned word from the activation. The read position is the end of line one and the target is fixed by line two, which comes later, so at that position the word can only be a likely rhyme. A control word matched on frequency is not matched on how natural a rhyme partner it is ("wheat" and "heat"), and a reader that only knows the last word could prefer the same partner the model does. The prompt-only floor at the current position would separate these; it is not measured.
- **Limits.** NLA: 1 sample per prompt, 1 layer. Oracle lens: 2 of its trained layers, where the benchmark passes an item at any layer, so 15 is a lower bound on its benchmark score. One control scheme, chosen after looking.


### CW-20: the rebuild experiment for the oracle lens (2026-10-08/09, exploratory)

Design: [designs/CW-20_olens_rebuild.md](designs/CW-20_olens_rebuild.md) (labels and the change of score recorded there before any variant was scored). Asked for by the project's TA. Outputs: [runs/cw20/](runs/cw20/) (the reconstructor's raw outputs, 420 MB, are not in the repo); analysis [runs/cw20/analysis.txt](runs/cw20/analysis.txt); figure `figs/cw20_olens_rebuild_by_kind.png`.

Qwen3.6-27B bf16, layer 44 (a trained layer for the oracle lens), the 70 multihop prompts of CW-14/CW-16b, 2 oracle lens write-ups per prompt (140). Reconstructor: `agu18dec/olens_and_ar` `ar_ptag_pooled`, loaded with the collaborator's `ar_vectors.py` at his commit b8d458e, imported unchanged; the TA read that loader and said to use it as is. Each bullet is fed on its own without its list mark and the predictions averaged over the write-up. Score: FVE in the reconstructor's whitened space around the corpus mean, with one global scale (0.43) fitted on the unedited write-ups. Pieces cut and labelled by Claude Sonnet, one call per write-up, 140 of 140 parsed. This run used torch 2.9.1 (cu128) where earlier runs used the benchmark's default build.

- **Checks.** The benchmark's layer-44 capture matches the block-44 output the loader's convention targets (cosine min 0.9997, 70 prompts). Unedited write-ups: FVE +0.21, mean cosine 0.46, the right prompt's activation retrieved among the 70 for 84% of write-ups; another prompt's write-up +0.02. Whole write-up as one text: +0.02. Published: 0.22 for the reconstructor on true text at L44, 0.487 "joint" for the oracle lens (definition not public).
- **What a write-up is made of** (share of characters, 140 write-ups): true statements about the prompt's subject 46%, invented follow-on questions and entities 24%, correct answer 10%, false statements about the subject 8%, markup 7%, wrong answer or placeholder 5%. Wrong text is 13%, against 33% for the NLA (CW-16b).
- **Measured** (paired over prompts, change in FVE points against 3 random removals of the same length, 95% bootstrap): removing the answer −7.7 [−9.2, −6.2]; removing statements about the subject +2.0 [+0.6, +3.4]; removing invented follow-ons +1.8 [+0.9, +2.7]; removing only the wrong pieces −0.2 [−1.5, +1.0] (66 prompts); removing markup 0.0 [−0.7, +0.6]. Correcting the wrong pieces in place: −0.3 [−0.7, +0.2]. Mean cosine gives the same signs.
- **Taken to mean.** The rebuild leans on the answer at the start of each bullet; invented follow-ons and statements about the subject carry slightly less than average text; the wrong pieces carry about what random text does and correcting them changes nothing. For the NLA the rebuild leaned on restatements of the prompt and correcting gained 5 points; the two reconstructors are trained on different things (the oracle lens's on the text that follows a token, the NLA's on a description), so the two results are about each reader-reconstructor pair.
- **Label validation.** Gemini 3.8 Flash labelled the same pieces of 30 random write-ups per reader, blind: oracle lens 426 pieces, agreement 89%, kappa 0.87; NLA (CW-16b labels) 380 pieces, 87%, kappa 0.84 (`scripts/cw20_validate.py`, `runs/cw20/validate/`).
- **Limits and the likely objection.** The score's centre was changed after the first check (the design's batch-centred score was negative, −0.14; the design says why the corpus centre is the reconstructor's own definition), so the +0.21 level is post hoc; the paired differences use the same score throughout. The answer sits at the start of a bullet and the random removals fall anywhere, so "answer" and "start of the bullet" are not separated. Our +0.21 is below the published 0.487, whose definition we could not check. One family, one layer, one reconstructor, labels by a model. The NLA's 41% (CW-16b) is centred on the 70 targets and is not comparable with this 21%.

## Limits

- One model. One sample per NLA cell in the pilot (CW-11: 4, CW-12: 2). Dev items only; nothing held out.
- multihop scoring is word match. Judges are unvalidated.
- No layer is a trained layer for both readers (Setup). L42 comparisons put the oracle lens off its trained layers; the sweep and L56 put the NLA off its.
- Number bridges: 8 of the 100 multihop items have a number as the bridge, and 5 list it in one form only (`12`, `7`, `8`, `11`, `three`), so a readout that gives the other form is scored as a miss. Seen once in a sample of 21 readouts ("3" for `three`). Not rescored.
- CW-12's random control is one draw (seed 0). The NLA-minus-oracle difference on a random part (35% against 17%) rests on that draw. The oracle lens's readouts on the random part are mostly unrelated text, and 72% of them contain Chinese against 30% on the whole activation, so its low score there may be a failure on unusual input.
- The oracle lens writes some Chinese in 30–72% of readouts (the NLA in none) and word match reads only Latin letters and digits. In 21 random Chinese-containing readouts scored as misses (CW-12; whole, the rest, random part), none names the bridge in Chinese.
- CW-12's 30 items are harder than the 50 they come from: on the same 30, CW-11's intact rates are 0.48 (oracle) and 0.44 (NLA), against 0.67 and 0.64 on all 50.
- No reader-free label for whether the model uses the hidden step (the collaborator's half addresses this).
- No SAE: no labelled SAE exists for Qwen3.6-27B (Gemma 3 has Gemma Scope 2).

## Next

1. Done as CW-10, CW-11 and CW-12. Still open: a removal that does not depend on J-Lens (reader-free erasure of the bridge, fitted on other items), with a probe to confirm nothing is left, then the same readers at 4 samples. That is what can separate read from re-solved on multihop.
2. Planted concepts split into J-space and non-J parts (as Viswanath did on Llama-3.3-70B), read by all three readers. This tests "outside the workspace" directly. Run on the collaborator's side.
3. Role-aware scoring of the multihop readouts with a validated role judge.
4. A probe-based presence measure for the bridge per layer.

## Collaboration and resources

This pilot is one half of a joint MATS 12 mini project with Vishesh Gupta (GitHub `senku14x`, a collaborator on this repo). His half builds reader-free causal references (which positions and contents the answer depends on) and ran the planted-concept experiments. Rows marked "Vishesh" in the plain summary are his results, stated in words only; the numbers and methods are in his materials.

- His raw outputs (public): Hugging Face dataset [`senku21x/causal-concordance-outputs`](https://huggingface.co/datasets/senku21x/causal-concordance-outputs).
- His code and design specs: `github.com/senku14x/Causal-Concordance` (private; ask him for access).
- Model: `Qwen/Qwen3.6-27B`. Benchmark: [`camilablank/workspace-bench`](https://github.com/camilablank/workspace-bench).
- Readers: J-Lens `neuronpedia/jacobian-lens` (Salesforce-wikitext, n1000); R-lens and the template lens `camilablank/workspace-lenses` @ d740106d; oracle lens `agu18dec/local-workspace` (`ckpts/ao/rl/s3d.ddp600.s0/iter_000600`); NLA `ceselder/qwen3.6-27b-nla-rl` (`av_base` + `av_rl_adapters/iter_000400`).
- The J basis used in CW-12 and how it is built: `jbasis/` in this repo.

## Prior work

- Viswanath, "Models are blind outside the J-space. NLAs aren't." (LessWrong, 2026-07-08): on Llama-3.3-70B, an injected concept's non-J part is invisible to the model and read by the NLA. Injection setup; does not cover natural prompts or re-solving.
- Bhatia, Blank, Nanda, "Towards surfacing model algorithms with meta-tokens in the J-space" (LessWrong, 2026-07-20): J-Lens readouts on Qwen3.6-27B carry process tokens. Context for J-Lens showing task words, not the hidden step, at mid layers.
- Prabhu, "Can you hide from a natural language autoencoder?" (LessWrong, 2026-06-24): NLA explanations on Qwen2.5-7B can be flipped while behaviour is unchanged.
- WorkspaceBench post (Nanda et al.): single-token lenses are bag-of-words; NLAs surface workspace content and hallucinate; oracle lens more trustworthy.

## Code

[scripts/](scripts/): `smoke_pass_a.py` and `smoke_pass_b.py` (lenses and oracle lens, then NLA); `ablate_pass.py`, `ablate2_pass.py` (deletion and swap); `chain_pass.py` (last-token reads for chain and brew); `classify_claims.py`, `judge_other.py`, `judge_roles.py`, `score_chain.py` (scoring); `patch_jlens_mem.py` (memory fixes to the WorkspaceBench J-Lens and R-lens code; scores are bf16 after it); `pod_bootstrap.sh`, `pod_sync.sh`, `*_chain.sh` (running on a rented GPU).
