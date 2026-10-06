# NLA vs lenses

What do activation readers tell us about Qwen3.6-27B, and what does a natural language autoencoder (NLA) capture that the oracle lens and J-Lens do not? A MATS 12 mini project, October 2026, on WorkspaceBench prompts.

**Everything is in [STATUS.md](STATUS.md):** a plain summary table at the top, then each experiment with its numbers, sample sizes, intervals and limits. Most results are exploratory; the one confirmed with tests fixed in advance is marked.

## Main results so far

- **On multihop prompts, both the oracle lens and the NLA name the hidden reasoning step from the "J part" of the activation** (its projection on the top 1,024 J-Lens directions, 16% of the squared norm) and rarely from the rest. Confirmed on 70 prompts not used before, with the tests fixed in advance. Plot: `figs/cw14_split_70_new_prompts.png`.
- **We have not found a clear NLA advantage** on hidden steps.
- **What the readers write differs.** Both carry about the same task content; the NLA adds a description of the prompt's form. From outside the J part the NLA still reports the prompt's last word correctly but gets the content wrong. Plot: `figs/cw15_content_by_part.png`.
- **Removing false statements from an NLA write-up does not improve what its reconstructor rebuilds**; it lowers it about as much as deleting random sentences of the same length.

## Where things are

| Folder | What |
|---|---|
| `STATUS.md` | The one account of the project |
| `designs/` | Plans written before each run (CW-10 to CW-16) |
| `scripts/` | All code. `cwNN_*` files belong to experiment CW-NN |
| `jobs/` | Job files run by `scripts/worker.py` on the GPU (CW-12, CW-14) |
| `runs/` | Raw reader outputs, saved activations and analysis outputs, one folder per experiment |
| `figs/` | Plots |
| `jbasis/` | The J-Lens basis used for the split, and the script that builds it |
| `to_read/` | Random examples of reader output, for reading by hand |

## Experiments

| | Question | Raw data |
|---|---|---|
| Pilot | Do the readers name a hidden step, and how often, across layers and tasks? | `runs/multihop100`, `runs/sweep50`, `runs/chain60*`, `runs/brew50*` |
| CW-10 | Does J-Lens show the step at another layer? Do write-ups that rebuild the prompt name the step more? | `runs/cw10` |
| CW-11 | Does deleting J-Lens directions hide content from the readers? | `runs/cw11` |
| CW-12 | What does each reader get from the J part and from the rest? (30 prompts) | `runs/cw12` |
| CW-13 | Where does the step rank in J-Lens's full list? Is it in the rest (template lens)? | `runs/jrank` |
| CW-14 | CW-12 repeated on 70 new prompts with tests fixed in advance | `runs/cw14` |
| CW-15 | What do the readers actually write from each part? | `runs/cw15` |
| CW-16 | What does the NLA's text rebuild through its reconstructor? | `runs/cw16` |

## Useful pieces of code

- `scripts/worker.py`: keeps one model loaded on a GPU and runs job files as they arrive.
- `scripts/cw16_ar.py`: runs the NLA's reconstructor (`ceselder/qwen3.6-27b-nla-rl`, `ar_reconstructor/`) on a list of texts and saves the rebuilt layer-42 activations.
- `jbasis/make_jbasis.py`: builds the J-Lens basis.

Joint project with Vishesh Gupta (`senku14x`). His results, data and the shared reader checkpoints are listed under "Collaboration and resources" in STATUS.md.
