# CW-16: what does the NLA's text rebuild? (design, 2026-10-06; exploratory)

Suggested by the project's TA (2026-10-06): remove the hallucinations from an NLA write-up, pass it through the reconstructor again, and see whether the rebuilt activation projects more onto the J part and whether FVE is higher. Written before the run.

## Setup

- The NLA is an autoencoder. Its verbalizer wrote the write-ups we already have. Its reconstructor (`ceselder/qwen3.6-27b-nla-rl`, `ar_reconstructor/`: a 43-block Qwen3.6-27B backbone and a linear head) maps a text back to a layer-42 activation. It reads `Summary of the following text: <text>{explanation}</text> <summary>` and predicts at the last token. Prediction and target are both scaled to norm sqrt(5120) before the error is taken, so it rebuilds direction, not size. Forward pass re-implemented from EasyNLA's `critic_predict` (read, not run).
- Texts: CW-14's NLA write-ups (70 multihop prompts, 2 samples, L42), no new verbalizer run. Targets: the saved activations.
- No reader is run. One model load, forward passes only.

## Text versions

1. **Original** write-up, target = the whole activation.
2. **Another prompt's** write-up (fixed derangement, seed 0). Floor.
3. **The prompt itself** as the text. Reference for "the text says exactly what was read".
4. **Hidden step swapped**: the hidden-step word replaced by another prompt's hidden step (only write-ups that name it).
5. **Hallucinations removed** (the TA's suggestion): an LLM rewrites the write-up minimally, keeping only statements consistent with the prompt. The rewriting model sees the prompt and the write-up.
6. **Content only** and **form only**: the same LLM call returns the write-up cut down to what the sentence is about, and to how it is written (format, structure, where it stops).
7. **Write-ups made from parts** (J part only, rest only, random part, …, from CW-14): does text read from a part rebuild that part?

## Measures

For a rebuilt vector p and target g, both scaled as the reconstructor's loss does, with μ the mean of the 70 scaled targets:
- cosine(p, g), and FVE = 1 − |p − g|² / |g − μ|² (mean over prompts);
- the same inside the J part (projection on the CW-12 basis, 1,024 directions) and inside the rest;
- share of p's squared norm in the J part (the targets' is 0.16);
- J-Lens rank of the hidden step on p (and of the swapped-in step for version 4), on CPU.

## What each outcome would say

- Removing hallucinations raises FVE: the false statements were costing reconstruction, so they are noise for the autoencoder. If it lowers FVE: the reconstructor uses them, i.e. they carry information about the activation even though they are false about the prompt.
- Content-only text rebuilds the J part and form-only text rebuilds the rest: the split seen in CW-15 holds in the other direction too. If both rebuild both parts about equally, CW-15's reading is weaker than it looked.
- Swapping the hidden step moves the J part toward the other prompt's and changes the J-Lens rank: the reconstructor places that word in the J part.
- If the original write-up does not beat another prompt's write-up clearly, the reconstructor is not working on these prompts and nothing else here is interpretable. This is the gate.

## Limits known in advance

FVE against the mean of 70 similar prompts is a harder baseline than the one the NLA was trained with (77% on web text), so absolute values will be lower. The reconstructor is the end-of-run checkpoint and the write-ups come from the step-400 verbalizer (its README says to pair any verbalizer with this reconstructor). LLM rewrites are one model, one prompt, not validated. Multihop only.

