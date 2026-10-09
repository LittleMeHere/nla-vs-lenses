# CW-20: the rebuild experiment for the oracle lens (written 2026-10-08, before the run)

Asked for by the TA (2026-10-08): the hallucination comparison with FVE, done for the oracle lens as it was for the NLA (CW-16b). The collaborator will run his own version separately; results to be compared.

**Question.** When the oracle lens's write-up is rebuilt into an activation by its own reconstructor, which parts of the write-up does the rebuild depend on, and does correcting the wrong parts raise it?

**Material.** The 70 multihop prompts of CW-14 and CW-16b (items 31–100), Qwen3.6-27B. Layer 44 (a trained layer for the oracle lens; CW-14 used 42). Activation at the bank's read token, captured with the benchmark's `backend.capture`. Oracle lens through the benchmark's `OLens`, 2 write-ups per prompt.

**Reconstructor.** `agu18dec/olens_and_ar`, `ar_ptag_pooled`, loaded with the collaborator's `scripts/011_olens_swaps/ar_vectors.py` at his commit b8d458e, imported unchanged (`attach_ar`, `ar_forward`, `whitener`). The TA looked at this loader on 2026-10-08 and said to use it as is. His accepted convention: input = layer tag + text, read `ln_pre`, raw output, scored as W(ŷ − μ) against W(h − μ) with the reconstructor's own whitener (`whitening_iolens_pooled_L44`); target = block-44 output.

**Score.** Whitened FVE after one global least-squares scale (his `scaled_fve`; the TA: "the loss ignores scale"), mean taken over the 70 targets. Mean per-sample cosine reported beside it.

**Checks before any variant is scored.**
1. The benchmark's layer-44 capture equals the block-44 output his code hooks (cosine > 0.999 on all 70).
2. How the write-up is fed. The published score is a "joint bullet FVE" of 0.487 whose definition is not public. Two readings are scored on the unedited write-ups: (a) the whole write-up as one text; (b) each bullet as its own text, predictions averaged. The one with the higher scaled FVE is used for the variants; both are reported. If neither is above 0.10 the run stops there.
3. Another prompt's write-up scored against this prompt's activation (the floor).

**Variants (after reading 10 write-ups and fixing the labels).** The oracle lens writes bullets that continue the prompt, so the NLA's labels do not carry over. Labels are fixed after reading, and recorded here before the reconstructor sees any variant. Planned shape, as in CW-16b: one model call per write-up cuts it into pieces that reproduce it exactly and labels each; versions are built by code; every removal has 3 random removals of the same length as control; a corrected-in-place version of the same length.

**Tests.** Paired differences over prompts with 95% bootstrap intervals, as CW-16b. Exploratory: labels by a model, one prompt family, one layer.

**Cost.** One A100 80 GB at $1.59/h, about 2 hours: about $3–4. Basis: CW-19's oracle pass (200 write-ups in about an hour including the model download) and CW-16b's reconstructor pass (3,376 texts in under 15 minutes).

## Labels (fixed 2026-10-08 after reading 10 write-ups, before the reconstructor saw any variant)

Each oracle lens bullet continues the prompt: usually an answer, then a follow-on (a new question, a comment, a restatement). Pieces are labelled:

- `answer_ok`: gives the correct continuation of the prompt (also in another language or inside brackets).
- `answer_wrong`: gives a wrong answer or an unfilled placeholder ("[State Name]").
- `about_ok`: restates or says something true about this prompt's own subject (its question, its entities, the hidden step).
- `about_wrong`: the same kind of thing but false or changed in meaning.
- `invented`: new questions, entities or scenarios the prompt does not contain.
- `format`: markup and framing with no content ("Answer:", "(Clue:", list marks).

Corrected version: `answer_wrong` and `about_wrong` pieces rewritten in place to be true, same length, nothing else changed. Removal groups: answer (ok + wrong), wrong (answer_wrong + about_wrong), about (ok + wrong), invented, format; each with 3 random removals of the same length.

## Check results and one change to the score (2026-10-09, before any variant was scored)

- Check 1 passed: capture cosine min 0.99966.
- Check 2, as written above (centre = mean of the 70 targets): whole write-up −0.43, each bullet with its list mark −0.34, each bullet without it −0.14. By the rule above the run would stop.
- The centre was mis-specified. The reconstructor's published FVE is measured in its whitened space around the corpus mean, not around the mean of a batch of similar prompts; our 70 targets share 31% of their whitened energy, which the batch centre removes from the denominator. With the corpus centre and one global scale: whole write-up +0.02, bullet with list mark +0.08, bullet without it **+0.22** (cosine 0.46, retrieval@1 among the 70 targets 0.84; another prompt's write-up +0.02). Published: 0.22 for the reconstructor on true text at L44, 0.487 "joint" for the oracle lens on held-out data.
- Decision: continue with each bullet fed on its own without its list mark, predictions averaged over a write-up's bullets, corpus-centred scaled FVE as the main score; the batch-centred score and mean cosine are reported beside it. This choice of centre was made after seeing the check, so it is post hoc.
