# G9 (#156): downsizing analysis and go/no-go for a 0.6B implementation

Date: 2026-09-14. Written analysis only — no 0.6B code, per #156's own
constraint. Inputs: G3/G4 (base 8B), G7 (fine-tuned 8B), `rlm_slice/README.md`
(prior 0.6B capability finding).

## Identifying the minimal sub-skill that produced the gain

Four arms were measured at both phases. The question is which one isolates
the *smallest* capability actually responsible for the accuracy gain.

| Arm | What it needs from the model | Base 8B | Fine-tuned 8B |
| --- | --- | --- | --- |
| A (bare 2-pass) | Nothing beyond baseline reasoning | 0.372 | 0.584 |
| R (controller-owned retrieval) | Nothing agentic — the *controller* always calls `retrieve()`; the model just reads injected context | 0.540 | 0.726 |
| B (real RLM, model-driven REPL) | The model must *decide* to open the REPL, write code, read output, iterate | 0.699 | 0.788 |

The critical comparison is R vs B, because R is the "retrieval alone, no
agentic decision-making" version of the same idea B implements end-to-end.

- **Base 8B**: R captures 77.2% of B's absolute score (0.540 / 0.699), and
  25.0% of B's total lift over the G2 zero-shot baseline (+5.3pp of B's
  +21.2pp). B still adds a real, substantial increment here (+15.9pp over R).
- **Fine-tuned 8B**: R captures 92.1% of B's absolute score (0.726 / 0.788),
  and 79.4% of B's total lift over baseline (+23.9pp of B's +30.1pp). The
  gap between "controller just retrieves for you" and "model runs the full
  agentic REPL loop" shrank from 15.9pp to 6.2pp.

**The minimal sub-skill is retrieval/context-augmentation, not agentic
tool-invocation.** After fine-tuning, nearly all of the RLM scaffold's value
is already present in the cheapest, least agentic arm. The model deciding
*for itself* to open a REPL and iterate contributes a real but shrinking
increment (6.2pp, down from 15.9pp) on top of that.

This lines up with cost, too: R runs at 1 decode call / ~7.3-7.8s per item;
B runs at ~10 decode calls / 58-84s per item — an 8-11x cost difference for
an increment that fine-tuning has already been compressing.

## Path A vs Path B, read against that finding

**Path A — distill the 8B RLM (Arm B) behavior into the 0.6B champion.**
Targets the harder, more expensive, and (per the above) proportionally
*smaller* remaining increment. It also faces a headwind this project has
already measured directly: `rlm_slice/README.md` documents that the 0.6B
champion could not self-initiate a single REPL block across every prompted
probe tried ("not reachable by prompting alone"). SFT distillation is a
different mechanism than prompting and could plausibly succeed where
prompting failed — Epic F's own distillation track already showed a 0.6B
can absorb an 8B teacher's MCQ-answering behavior — but this specific
sub-skill (deciding, unprompted, to invoke a code-execution tool mid-turn)
is a qualitatively different thing to imitate than "answer this MCQ," and
nothing measured so far establishes that it survives 0.6B-scale distillation.
Unproven, and aimed at the smaller remaining prize.

**Path B — extract retrieval as one fixed, Cactus-safe step.** Requires
*no* new capability from the 0.6B at all: the app (not the model) decides
to retrieve, exactly as `rlm_slice/retrieval.py`'s `retrieve()` already does
for Arm R — the model only has to read injected context and answer, which
is the same shape of task the 0.6B champion already does today. Captures
the large majority of the measured gain (92.1% of B's score, 79.4% of B's
lift, on the fine-tuned model) at a fraction of the latency cost, and sidesteps
the exact capability gap (agentic tool-initiation) this project has already
found the 0.6B lacks.

**Path B is not free of engineering work, though.** `retrieval.py`'s
`retrieve()` shells out to a `qmd` subprocess against QMD collections running
on this dev machine — that is a desktop-side tool, not something that runs
inside a Cactus on-device bundle. Making this genuinely "Cactus-safe" needs
one of: (a) a network call to a hosted retrieval endpoint at inference time
(the app already has some network dependency, e.g. Supabase logging, so this
is a narrower gap than it sounds, but changes the offline story), or (b) a
real on-device index (embedding/BM25 index bundled with the app) — a
separate, nontrivial scoping question of its own.

This is also where the QMD-collection question raised at the very start of
this epic's work becomes directly relevant again: the deferred plan to build
a consolidated "all categories" QMD collection (rather than the current
per-category-limited collections) is now a concrete prerequisite for Path B,
not a nice-to-have — a fixed retrieval step is only as good as the corpus it
searches, and a category-limited collection would silently cap what Arm-R-style
retrieval can find on-device.

## Go/no-go decision

**GO on Path B (retrieval-as-fixed-step) as the near-term downsizing target.
NO-GO on Path A (full RLM distillation into the 0.6B) for now** — not ruled
out permanently, but not the next thing to build, given it targets a smaller
proportion of the gain, faces an unproven capability-transfer question this
project has independent evidence to be skeptical of, and costs far more at
inference time even if it did transfer.

Concrete next steps this decision unblocks (still no 0.6B training/inference
code — these are scoping/infra questions Path B needs answered first):
1. Resolve how retrieval executes at on-device inference time (network vs.
   bundled index) — a deployment-architecture decision, not a training one.
2. Revisit the deferred QMD "all-categories" collection question with this
   concrete use case in hand, rather than as a speculative nice-to-have.
3. Only after (1)-(2): a scoped follow-up ticket to actually wire fixed
   retrieval into the 0.6B champion's inference path and re-run the F9-style
   on-device ladder to measure the real gain on the deployed model, on the
   deployed hardware, before any 0.6B-side model changes are considered.

## Accept criterion (#156)

Written transfer analysis (above) plus a recorded decision: **GO on Path B,
NO-GO on Path A for now.** No 0.6B code precedes this analysis, per the
ticket's own constraint. This closes Epic G's active work (M1-M4); G8 (#155)
was skipped as infeasible on available hardware, recorded separately.
