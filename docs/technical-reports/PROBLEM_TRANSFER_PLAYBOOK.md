# Problem Transfer Playbook

This playbook explains how to move the solver-synthesis recipe from one optimization problem to another without pretending the recipe is fully plug-and-play.

The core lesson from SDS, the JSSP transfer, and the reward-normalization ablation is:

- the outer scaffold transfers better than the exact reward calibration
- the risky part is not the GRPO loop itself
- the risky part is the task-reward geometry and solver contract for the new domain

In other words: move the architecture, then re-calibrate the reward geometry.

## 1. What should transfer unchanged

These pieces are the most portable and should be preserved unless there is a strong reason to change them.

- the compiler-style framing: generate one standalone solver artifact rather than per-instance chain-of-thought answers
- the GRPO training loop and group-based sampling regime
- the three-part reward-stack shape:
  - format reward
  - execution / feasibility reward
  - nominal objective-quality reward
- the public evaluation philosophy:
  - feasibility first
  - then objective quality relative to native baselines and a domain-specific VBS
- the prompt philosophy:
  - define the contract clearly
  - ask for a reusable white-box solver
  - discourage brittle one-pass heuristics on hard instances

## 2. What must be re-derived per problem

These pieces should be treated as domain-specific by default.

- the solver input/output contract
- the simulator and feasibility checker
- the nominal reward normalization
- any light semantic shaping
- the native baseline set
- the failure taxonomy used for logging and analysis

Do not assume that a reward component that worked on SDS is automatically safe on CVRP, JSSP, or another domain.

## 3. Recommended porting order

### Step 1: Freeze the public solver contract

Define exactly what a legal solver must return.

Examples:

- SDS: selected variable set
- JSSP: valid `job_sequence`
- CVRP: route set with depot closure, customer coverage, and capacity satisfaction

If the contract is fuzzy, the reward will be fuzzy and the model will exploit that ambiguity.

### Step 2: Build the execution reward before the nominal reward

First make sure the model can reliably produce code that:

- parses
- runs
- emits the correct schema
- returns a candidate solution object
- passes hard feasibility checks

Only after this is stable should the nominal objective-quality signal become the main optimization target.

### Step 3: Start with a brutally simple nominal reward

Do not begin with a clever normalization heuristic.

Start with a simple, inspectable signal such as:

- feasible route cost
- feasible makespan quality
- feasible SDS score

Then add normalization only after you have inspected the raw scale and variance.

### Step 4: Calibrate normalization empirically

This is the biggest lesson from the SDS normalization ablation.

For a new domain, compare a small set of candidate normalizations early, for example:

- fixed naive bound
- optimistic top-k bound
- simple baseline-relative normalization
- clipped instance-relative normalization from sampled feasible solutions

Evaluate them on:

- seed stability
- feasibility rate
- final mean gap
- reward-scale pathologies

Do this before committing to a large training campaign.

### Step 5: Keep feasibility-first pressure early

For combinatorial domains, it is usually safer to learn legality first and refinement second.

That means:

- strong positive signal for legal outputs
- weak or zero nominal credit for infeasible outputs early on
- only soften this later if there is clear evidence the hard gate is too destructive

If you soften too early, the model often learns to chase attractive but illegal structures.

### Step 6: Add only minimal semantic shaping

Use semantic shaping only when the public contract is easy to fake syntactically.

Good shaping is:

- minimal
- interpretable
- tied to the contract
- easy to explain publicly

Examples:

- JSSP: operation progression and machine/job readiness semantics
- CVRP: route closure, customer coverage, capacity preservation, and non-degenerate local improvement structure

Avoid hidden imitation targets or domain-specific hacks that are hard to justify.

### Step 7: Instrument failures from day one

Do not wait until a appendix to understand failure modes.

For a new domain, log separate buckets for:

- malformed output
- runtime error
- timeout
- infeasible output
- feasibility subtype
- low-quality but feasible output

For CVRP specifically, separate at least:

- missing customers
- duplicated customers
- depot/route closure failure
- capacity violation
- empty or degenerate routes

### Step 8: Validate compile-once early

Do not wait until the end of the project to test whether the learned artifact is really reusable.

Generate a frozen solver and run it unchanged across a held-out set.

This tells you whether you are learning:

- a reusable algorithm

or

- prompt-conditioned per-instance patching

### Step 9: Use one strong native baseline and one same-family manual baseline

For interpretation, you usually want both:

- a strong classical/native baseline
- a hand-written baseline from the same algorithm family the model appears to learn

For CVRP, that might mean:

- a strong OR-Tools or local-search baseline
- one hand-written route-improvement heuristic

This helps distinguish genuine solver synthesis from mere recovery of an obvious textbook family.

### Step 10: Run a small transfer stress test before scaling

Before expensive training, do a small campaign:

- 100 to 300 instances
- 2 to 3 seeds
- 2 to 4 reward variants
- one frozen-solver evaluation

If this small study is brittle, scaling up usually just burns compute faster.

## 4. What to tell someone porting the recipe to CVRP

If someone wanted to adapt the recipe to CVRP now, the advice would be:

1. lock the route contract first
2. make the execution reward brutally reliable
3. start with a simple feasible-cost signal
4. compare several normalization schemes early
5. log capacity / coverage / route-structure failure types explicitly
6. validate one frozen solver before spending heavily on more training
7. treat reward normalization as a calibrated design choice, not a reusable default

The correct mindset is not:

- copy SDS and hope

It is:

- preserve the scaffold, then re-fit the task reward and contract carefully

## 5. Recommended claim discipline

For future papers or branches, the safe claim is:

- the solver-synthesis scaffold can transfer across optimization problems
- but the simulator, contract, and reward calibration remain problem-specific

That is stronger and more honest than claiming a domain-agnostic recipe, and it is better supported by the evidence we currently have.
