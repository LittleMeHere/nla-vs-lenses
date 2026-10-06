# CW-14: confirming run of the J part / rest split on prompts not used before (design, 2026-10-05)

Written and committed before the run.

## Why

CW-12 (30 prompts) found that the oracle lens and the NLA name the hidden step from the J part about as often as from the whole activation, and much less often from the rest. The cutoff (1,024 directions), the conditions shown and the scoring were chosen while looking at those 30 prompts, so the result is exploratory. This run repeats it with nothing changed, on prompts that were not in CW-12.

## What is fixed in advance

- **Prompts.** The 70 multihop items of the 100-item bank that were not in CW-12: items 31 to 100 in the bank's order. Items 31 to 50 were used in CW-11 (cue removal, a different manipulation); none were used in any split. The bank is ordered by item type, so these 70 are mostly different item types from the first 30.
- **Model and layer.** Qwen3.6-27B bf16, layer 42, the bank's read position. Activations are captured fresh; the first 30 are also captured and compared with the saved CW-12 activations as a check on the capture.
- **Conditions (4).** whole; J part (projection on the top 1,024 directions of the saved CW-12 J basis, `runs/cw12/jbasis.pt`); rest (whole minus J part); random part (the same random 1,024-dimensional subspace as CW-12, seed 0).
- **Readers.** Oracle lens and NLA, 2 samples per condition, seeds set per item and condition as in CW-12.
- **Score.** Share of a prompt's samples that name the hidden step, same word match as CW-12 (`names()` in `scripts/cw12_analyze.py`). Unit of analysis: the prompt.

## Primary tests

Paired difference over prompts, two-sided sign-flip permutation test (100,000 draws) and 95% bootstrap interval.

1. Oracle lens: J part minus rest.
2. NLA: J part minus rest.
3. Oracle lens: J part minus random part.

A test counts as confirmed at p < 0.005 with the CW-12 sign. It counts as not confirmed if the 95% interval includes 0. Anything between is reported as weak.

## Sample size

CW-12 gave J minus rest = +0.28 with a 95% interval of [0.12, 0.47] at n = 30, and the SD of a paired difference over prompts is 0.50 to 0.52 (measured from the CW-12 rows; its sign-flip p values were 0.004 to 0.008). At n = 70 the SE is about 0.06. If the true difference is 0.28, the expected z is about 4.7 (about 93% chance of p < 0.001). If it is 0.20, z is about 3.4. 70 is every unused prompt in the bank.

## Secondary (reported, not used for the verdict)

- NLA: J part minus random part (CW-12: +0.07 [−0.08, 0.22], not shown).
- J part minus whole, both readers, with interval.
- Prompts where only one reader names the step on the whole activation (CW-12 and earlier: NLA-only 3, 3, 0; oracle-only 8, 6, 1).
- From the saved activations, on CPU: template lens and J-Lens rank of the hidden step in each part (CW-13's measures).

## Limits known in advance

One prompt family, one layer, one J basis, one cutoff, one random subspace, word match, 2 samples per cell. A part of an activation is off-distribution for both readers. This run can confirm CW-12's pattern on new prompts; it cannot show the pattern holds at other cutoffs or on other families (Vishesh's rerun covers those).


## Extension (added 2026-10-05 19:59 UTC, before any reader output existed)

Four more conditions on the same 70 prompts, same readers, 2 samples, run after the four main conditions. They are secondary: the three primary tests and their rule are unchanged.

- **All but the random part** (whole minus the seed-0 random part). Test: rest minus all-but-random, both readers. CW-12: −0.25 and −0.27. This is the claim that removing the J part costs more than removing a random part of the same size.
- **Second random part** (seed 1). CW-12 used one random subspace; this shows whether the random-part result depends on the draw.
- **PCA part and all but the PCA part** (top 1,024 centred PCA directions, the saved CW-12 basis `runs/cw12/pcabasis.pt`, projection without re-centring as in CW-12). Asked for by Vishesh in CW-12. Tests: J part minus PCA part; rest minus all-but-PCA.

The parts are built on CPU from the saved whole activations (`scripts/cw14_ext_cells.py`) and sent to both pods.

Not addressed: the NLA's J part minus random part (CW-12 +0.07, SD 0.43) would need about 300 prompts to detect at that size, and the bank has 70 unused.


## Add-on: does the model itself need the J part? (added 2026-10-05 20:45 UTC, exploratory, no reader involved)

Question: is the J part something the model works with, or only the part that is easy to turn into words? The readers cannot answer that. This add-on asks the model.

- For each of the 100 multihop prompts, run the base model with the layer-42 activation at the read position replaced by one part, and read the next-token distribution at the end of the prompt. Reference: the unpatched run's top token.
- Parts: J part only, rest only (J part removed), random part only and all but random (seeds 0 and 1), PCA part only and all but PCA, unembedding part only and all but it (top 1,024 principal directions of the unit unembedding rows, no Jacobian; `scripts/cw14_unembed_basis.py`; overlap with the J part 0.32 where 0.20 is unrelated). No rescaling.
- Measures: whether the top token is unchanged, and the change in log-probability of the unpatched top token. Paired over prompts; main contrast is removing the J part against removing a random part of the same size. Reported for the 70 fresh prompts and for all 100.
- Limits: one layer and one position are patched, so the model can still reach earlier layers of that position by attention from later tokens (33 prompts have one later token). The J directions come from the Jacobian to the output, so some effect of removing them on the output is expected by construction; the random and unembedding parts are the comparison. It tests the final answer, not the hidden step.
- About 1,100 forward passes on the loaded base model.

## Add-on: centred activations (added 2026-10-05 22:30 UTC at Vishesh's request; exploratory)

Vishesh's idea: most of an activation is a direction shared by all prompts, the readers rescale their input, so removing that direction may let them read more. His spec (message of 2026-10-05): centre by projecting out the unit direction of a mean, c = h − (h·u)u, with two means.

- **Task mean:** the mean L42 read-position activation of the other 99 multihop items (leave the current item out). Built on CPU (`scripts/cw14_centre_cells.py`).
- **Generic mean:** the mean L42 activation over about 200 wikitext passages, token positions after the first 16, no bank items. Captured on the base pod (`jobs/cw14_e_generic.py`). This is the one that could be used on any prompt.
- Same 70 prompts, oracle lens and NLA at L42, 2 samples, word match. Comparison: centred minus whole, per reader, paired over prompts.
- Known in advance: the oracle lens names the step from the whole activation in 86% of samples on these prompts, so it has little room to rise; L42 is off its trained layers. His own brew run (007 stage 1a, oracle by likelihood at L40/L44) found no gain from mean removal. Not included here: the J part and random parts of the centred activation, and likelihood scoring; his multihop run covers those for the oracle lens. The NLA is not in his run (its path failed a gate there), so the NLA rows are the new information.
