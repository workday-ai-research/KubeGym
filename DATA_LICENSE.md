# Data licence

The code in this repository is MIT-licensed (see `LICENSE`).

The Azure-derived part of `corpus/` is derived from the **Azure Functions Trace 2019**, published by
Microsoft under the CC-BY Attribution License. Source:
https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md

Attribution, as required by the dataset:

> M. Shahrad, R. Fonseca, I. Goiri, G. Chaudhry, P. Batum, J. Cooke, E. Laureano, C. Tresness,
> M. Russinovich, R. Bianchini. "Serverless in the Wild: Characterizing and Optimizing the
> Serverless Workload at a Large Cloud Provider." USENIX ATC 2020.

What we changed: per-minute invocation counts were summed from functions to applications, a
subset of applications was selected, and request-level traces were reconstructed from the
counts. The reconstruction rests on assumptions we made. It is not a property of the source
data. See `docs/WORKLOADS.md`.

The generated families (constant, variable, burst, diurnal) are synthetic, produced by
`kubegym/workloads/generator.py`.
