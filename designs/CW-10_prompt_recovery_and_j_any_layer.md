# CW-10: prompt recovery and the J-any-layer check (design, 2026-10-04)

Not run. No GPU and no new model calls: both analyses use readouts already saved in `runs/`. Exploratory: pilot readouts (1 NLA sample, 1 oracle readout per cell, word match, bf16), no dev/test split on multihop yet.

## Question

When the oracle lens or the NLA names the hidden step and J-Lens does not, did it work the step out from a prompt it rebuilt?

## Part A: J-any-layer check

- **Items.** The 36 multihop items where the oracle lens and the NLA name the bridge at L42 and J-Lens top-10 does not (CW-1).
- **Measure.** For each, does J-Lens top-10 name the bridge at any of its 11 saved layers (`runs/multihop100/jlens.multihop.jsonl`, same word match as CW-1)?
- **Report.** Count of the 36 named at some layer, with the layer; the same count for the other 25 non-J items as context.
- **Reading.** Items J-Lens never names at any layer are the strongest re-solving candidates. Items it names at a later layer are consistent with content that is present but not yet in J-Lens's top-10 at L42.
- **Already known, before this design:** J-Lens names the bridge at one or more layers in 76 of 100 items (39 at L42). The per-item breakdown for the 36 has not been computed.

## Part B: prompt recovery

- **Recovery score.** For one readout: the share of the prompt's content words that appear in it. Content words = lowercased word tokens of the item's prompt, minus a fixed stop list, minus the bridge and answer words (so the score cannot contain the outcome).
- **Outcome.** The readout names the bridge (CW word match).
- **Primary test.** The 50 items of CW-4, removal condition (`runs/abl3`, cos > 0.8 neighbourhood removed), oracle lens and NLA separately: AUC of the recovery score for naming the bridge after removal. 95% interval by bootstrap over items.
- **Controls.**
  - Readout length alone as the predictor (a longer readout has more chances at both).
  - Recovery scored against another item's prompt (chance level for the score).
  - The same test with the score taken from the intact L42 readout of the same item, so predictor and outcome come from different readouts.
- **Secondary.** The same AUC on the L28 readouts (CW-2, 50 items) and the intact L42 readouts (CW-1, 100 items).
- **What it can show at this size.** With 50 items and about 20 positives, the interval on an AUC is roughly ±0.15. An AUC near 0.65 will not be distinguishable from 0.5. Only a strong relation (about 0.75 or more) or its absence will be clear. A firm answer needs more items or more samples per cell, which needs a GPU run.

## Not in this design

- No judge: word match only, as in the pilot rows it reuses.
- Results are labelled CW-10 and exploratory.
- Multihop dev/test split: proposed grouping is by relation family (the item id prefix), to be agreed before any reported multihop number.

