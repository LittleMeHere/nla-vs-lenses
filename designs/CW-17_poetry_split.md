# CW-17: the J part / rest split on poetry, and where a held word shows up by layer (design, 2026-10-06)

Written before the run. Asked for by the collaborator (2026-10-06): run the split on families other than hidden reasoning steps.

## Poetry split

- **Prompts.** WorkspaceBench poetry, all 100 items. Each is a couplet with the second line cut before its last word; the content looked for is the rhyme word the model is planning (bank field `intermediates`, one word). Read at the newline that ends line one, the bank's read position.
- **Readers and layers.** NLA at layer 42 (its trained layer). Oracle lens at layer 44 (a trained layer for it; the TA's advice). Each reader gets parts built from its own layer's activation and its own layer's J basis: the saved L42 basis, and an L44 basis built the same way (`jobs17/cw17_a1.py`).
- **Conditions.** whole; J part (top 1,024 directions); rest; random part (seed 0). 2 samples per cell.
- **Score.** Share of a prompt's samples that contain the rhyme word (word match). Rhyme words are short common words, so a chance rate is reported: the same score against another prompt's rhyme word (fixed derangement).
- **Pre-set tests** (paired over prompts, sign-flip permutation, 95% bootstrap), for each reader: J part − rest, and J part − random part. Confirmed at p < 0.005 in the direction found on multihop (J part higher). If a reader names the rhyme word from the whole activation at or near the chance rate, its tests are reported as uninformative.
- **Also reported.** The same restricted to prompts where the reader names the word from the whole activation in both samples (the TA's gate). J-Lens top-10 per part.
- **Sample size.** CW-14's paired differences had SD 0.41–0.47 over prompts; at n = 100 the SE is about 0.045, so a difference of 0.2 would be about 4 SE.
- **Before the full run.** The first 5 prompts' write-ups for every condition are read as soon as they exist.

## Directed modulation: layer check only (no reader)

The collaborator's earlier study (Readable Here, Used There) found the held word readable by J-Lens mainly at layers 51–59, weakly at 48–50. The NLA reads only layer 42. Before any reader run, capture the benchmark's 100 directed-modulation items at every writing position, layers 20 to 60 in steps of 4 and layer 42, and compute on CPU the J-Lens rank of the concept word by layer and by instruction (think / do not think / suppress). No test; this decides where, if anywhere, a reader run is worth doing.

## Limits known in advance

One poetry bank, 100 items. The two readers are at different layers, so reader-against-reader differences are not interpretable here; each reader's own J part − rest contrast is. Word match on short words.
