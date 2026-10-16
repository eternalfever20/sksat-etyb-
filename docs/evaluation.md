# Evaluation

Run these from the repository root:

```text
python -m pytest -q
python -m evaluation.run_eval
```

The tests cover CSV and Excel ingestion, semantic typing, ten analysis tools, statistical results against known calculations, chart output, invalid inputs, tool-call iteration, provider error handling and persistent/scoped SQLite facts. The offline evaluation checks deterministic calculations against the synthetic sales sample and UCI Iris, a real dataset with a documented source and license. Agent unit tests use fake HTTP transport; they do not measure live language-model quality. An IMDb-shaped test fixture covers years and multi-value genre fields.

An optional live check is available with `python -m evaluation.run_eval --live`. It needs a working OpenRouter key, network access and an available model route. A provider failure means the live check could not establish model behavior; it does not change the offline test result.
