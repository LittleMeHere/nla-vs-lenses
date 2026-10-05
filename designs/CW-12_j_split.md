# CW-12: what do the readers get from the J-Lens part of an activation, and from the rest? (design, 2026-10-05)

Written before the run. Follows CW-11, which showed that deleting a few J-Lens directions does not hide content from the prose readers.

## Question

Neel's question on natural activations: is what the oracle lens and the NLA report in the part of the activation J-Lens reads, or outside it?

## Method

- **The J part.** At L42, token t's J-Lens direction is d_t = W_U[t] J, and J-Lens scores t by unit(d_t)·h. Take all vocabulary directions, unit-normalised, and their top-k principal directions (eigenvectors of Σ_t unit(d_t) unit(d_t)ᵀ). The J part of h is its projection onto those k directions; the rest is h minus that. This uses J-Lens's own map and no reader. The whole vocabulary spans all 5,120 dimensions, so k is a choice: it is swept, not fixed in advance.
- **Control.** A random k-dimensional subspace and its complement.
- **Readers.** J-Lens top-10, oracle lens, NLA, each given one part at a time. Both prose readers rescale their input, so each part arrives at the usual norm; the share of the activation's squared norm in each part is recorded.
- **Items.** The 50 multihop items of CW-4 and CW-11, L42, same activations.

## Steps

1. **Check on 5 items first** (the lesson of CW-11): k in {64, 256, 1024, 2560}, oracle lens only, 2 samples. Look at norm shares, whether J-Lens still shows its top tokens on the J part and loses them on the rest, and whether naming and prompt recovery move with k at all.
2. Pick one or two k from step 1 and state why. Then 30 to 50 items, both readers, parts J / rest / random / random-rest / full.

## Measures

Share of samples naming the bridge (word match, as CW-1) and prompt recovery (as CW-10), per part. Paired differences over items with bootstrap intervals.

## What each outcome would mean

- Readers name the bridge and recover the prompt from the rest about as well as from the full activation, and J-Lens shows nothing there: the prose readers read content outside what J-Lens reads. This is the "not in the workspace" direction of Neel's question, for reading only; it says nothing about whether the model uses that content.
- Readers need the J part: what they report sits where J-Lens reads, and the gap to J-Lens top-10 is about display or layer.
- Both parts suffice: the content is spread redundantly and this split cannot localise it.
- If the random split behaves like the J split, the result is about dimension count, not about J-Lens.

## Limits known in advance

"J part" is one construction among several. A part of an activation is off-distribution for readers trained on whole activations. Word match. 50 items, one prompt family, one layer.

