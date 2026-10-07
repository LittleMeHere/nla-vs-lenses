# CW-19 (base worker): poetry through the benchmark's own producer, oracle lens at its trained layers 40 and 44, default sampling.
p.use("olens")
p.run_family("poetry", "/workspace/out/cw19/olens_poetry.jsonl", layers=[40, 44])
print("CW19_OLENS_DONE", flush=True)
