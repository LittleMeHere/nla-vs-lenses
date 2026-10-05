# CW-11: cue removal with 4 samples per readout (design, 2026-10-04)

Follows CW-10. One GPU run on the 50 multihop items of CW-4, read at L42.

## Question

CW-10 found that NLA readouts which recover more of the prompt name the bridge more often. Two readings: (a) the reader rebuilds the prompt and works the bridge out; (b) a readout that reads the activation well contains both. CW-4 removed the bridge's directions and left the prompt. This run does the mirror: remove the prompt's cue words and leave the bridge.

## Conditions (per item, same L42 activation as CW-4)

| condition | removed from the activation |
|---|---|
| intact | nothing |
| bridge | the bridge's J-Lens directions and their cos > 0.8 neighbourhood (as CW-4) |
| cue | the J-Lens directions of the prompt's content words (bridge and answer words excluded) and their cos > 0.8 neighbourhood, minus any direction in the bridge set |
| both | bridge set and cue set |
| rand | as many random vocabulary directions as the cue set |

## Readers and samples

J-Lens top-10 (plus the bridge token's J-Lens score in every condition), oracle lens 4 samples, NLA 4 samples. Fixed seed per cell. NLA = `ceselder/qwen3.6-27b-nla-rl`, `iter_000400`.

## Measures

- Per item and condition: share of the 4 samples that name the bridge (word match, as CW-1), and the prompt-recovery score of each sample (as CW-10).
- Checks that the removal did what it should: under `cue`, the recovery score falls against `rand`, and J-Lens still scores the bridge as before; under `bridge`, J-Lens top-10 loses the bridge (as CW-4).
- Main contrast: naming rate under `cue` against `rand`, paired over the 50 items, bootstrap interval. Second: `both` against `bridge`.
- Also: the CW-10 AUC recomputed with naming as a rate over 4 samples, and a hand check of 30 random readouts against word match.

## Predictions (written before the run)

- If readers work the bridge out from the prompt: naming falls under `cue` even though the bridge's directions are untouched, and falls further under `both`.
- If they read the bridge: naming under `cue` stays at the `rand` level.
- If `cue` does not lower the recovery score, the removal failed to hide the prompt and the run says nothing about either reading.
- My guess: the recovery score falls only partly, and naming under `cue` falls by less than it did under `bridge` in CW-4 (34 → 23 of 50). Confidence low.

## Size and cost

250 cells. Earlier runs: NLA 6.4 s per single sample, oracle 8.2 s per readout, model loads about 5 min each, downloads about 25 min. Estimate 2.5 to 3.5 hours on one A100 80GB. With 50 items, a paired drop of about 15 points in naming rate is detectable; a smaller one is not.

## Limits known in advance

Word match counts mentions. J-Lens directions may not hold all of the prompt's information, so `cue` may be a weak removal (the recovery check measures this). Readers are the subject model plus a LoRA at their training layer. One prompt family, no dev/test split.

## Change after the first check (2026-10-04, before any reader output was looked at)

The J-Lens check on the built activations showed the removals work as intended on tokens (cue words in J-Lens top-50: 6.7 per item intact, 0.02 after cue removal; bridge in top-10: 14 intact, 0 after bridge removal). It also showed that plain cue removal lowers the bridge's own J-Lens score (mean 1.95 intact, 1.13 after random removal, 0.32 after cue removal), because cue and bridge directions are not orthogonal. A fall in naming under `cue` could therefore come from damage to the bridge.

Added a sixth condition, `cue_orth`: the cue directions are first made orthogonal to the bridge set's span, then removed, so every bridge direction keeps exactly its original coordinate. Same readers and samples. `cue_orth` against `rand` is now the main contrast; plain `cue` is kept and reported. Scripts: `cw11c_pass_a.py`, `cw11c_chain.sh`.
