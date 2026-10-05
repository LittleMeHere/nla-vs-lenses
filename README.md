# NLA vs lenses pilot

When a natural language autoencoder (NLA) or the oracle lens names a hidden reasoning step in Qwen3.6-27B, did it read that from the activation, or re-solve the question from the prompt it decodes? Compared against J-Lens on WorkspaceBench items.

- Summary and all results: [STATUS.md](STATUS.md)
- Code: `scripts/`. Raw readouts, items, logs and saved activations: `runs/<experiment>/`.
- Plots: `figs/`. Designs written before each run: `designs/`. Job files for the later runs: `jobs*/`.
- Latest results (4–5 October): `runs/cw11` (removing prompt words from the activation), `runs/cw12` (splitting each activation into the part J-Lens reads and the rest), `runs/jrank` (where the hidden step ranks in J-Lens's full list, and the template lens). Each has an `analysis.txt`.
- The J-Lens basis used for the split, and how it is built: `jbasis/`.

Exploratory pilot (MATS 12 mini project, October 2026).
