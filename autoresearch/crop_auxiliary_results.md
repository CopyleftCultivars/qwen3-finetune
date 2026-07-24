# CROP-style auxiliary-task pilot results

These are four independent fixed-dose pilots on the strict v5clean base. Each
candidate adds 256 auxiliary training rows and 64 auxiliary validation rows;
the remaining rows are unchanged v5clean training/validation data. The sealed
Spanish benchmark was not read or used.

The primary is the frozen English MCQ option-set-normalized gold logprob. The
v5clean anchor is `-0.7751536917` with argmax accuracy `0.7256637168`.

| Lever | Native validation change | Frozen English MCQ primary | MCQ argmax | Disposition |
|---|---|---:|---:|---|
| Summarization | token F1 `0.2553 -> 0.8749`; exact text `0.0% -> 29.7%` | `-1.1328990334` (`-0.3577453417`) | `65.49%` | Not promotable in this dose; closed-book regression |
| NER/entity extraction | token F1 `0.3635 -> 0.7262`; JSON validity `29.7% -> 68.8%` | `-0.6956284052` (`+0.0795252865`) | `75.22%` | Promising diagnostic, but below the `+0.4` primary promotion gate |
| Structured extraction | token F1 `0.7507 -> 0.9119`; JSON validity `40.6% -> 56.3%`; exact JSON `0.0% -> 9.4%` | `-0.8931188782` (`-0.1179651865`) | `69.91%` | Not promotable in this dose; closed-book regression |
| Domain open-book QA | token F1 `0.1194 -> 0.9497`; exact text `0.0% -> 92.2%` | `-1.1055118810` (`-0.3303581893`) | `59.29%` | Not promotable for the closed-book loop; retain as a separate grounded-task diagnostic |

The native gains show that all four task streams are learnable. They do not
show that the tasks should be mixed into the closed-book autoresearch recipe.
NER is the only candidate that improved the frozen MCQ primary, but its gain
is well below the existing promotion threshold. No combination experiment is
authorized by these results.

## Evaluation boundary

Completed for every candidate:

- native task validation against the task's held-out auxiliary validation set;
- frozen English MCQ regression against `data/val_v3.jsonl`;
- primary-objective record under `autoresearch/objective_weights.json`.

Not run:

- paired free-form rubric/grader-anchor guardrails, because that pass would
  transmit model-generated evaluation answers to an external judge and no such
  evaluation authorization was provided. The OpenRouter budget was untouched.

The ignored per-run JSON artifacts under `results/` retain the raw native and
MCQ measurements locally. This compact report is the committed comparison
record; generated model weights, composed overlays, and per-item evaluation
dumps remain ignored and reproducible.

