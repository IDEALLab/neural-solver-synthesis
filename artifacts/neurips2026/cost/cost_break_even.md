# N26-E3-v3 Full Cost and Break-Even Accounting

Analysis commit: `4a0eae0916934a7e0bcb247fe9f9085f435efee3`

## GPU break-even

- One-policy ordinary Hero: 11,239--11,598 deployment instances.
- One-policy compile-once Frozen Hero: 11,044--11,433 deployment instances.
- Full three-seed campaign, ordinary deployment: 33,707--34,802 instances.
- Full three-seed campaign, compile-once deployment: 33,121--34,307 instances.
- Scheduler-allocation sensitivity, one policy: ordinary 2,810--2,900; compile-once 2,766--2,863 instances.
- Scheduler-allocation sensitivity, full campaign: ordinary 8,427--8,701; compile-once 8,294--8,591 instances.

Ordinary Hero retains one completion per deployment; Frozen Hero charges the 64-completion validation synthesis once. CPU execution, active GPU generation, billed allocation, and wall time remain separate. ShinkaEvolve synthesis and billing remain unavailable.

The phase-by-seed ledger is `per_seed_costs.csv`; hashes and evidence definitions are in `provenance.json`.
