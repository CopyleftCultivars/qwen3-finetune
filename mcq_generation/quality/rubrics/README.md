# MCQ Quality Assessment Rubrics

This directory contains rubric templates for evaluating MCQ quality across 9 agricultural benchmark categories.

## Files

### Base Template
- **`rubric_base.md`** — Core 5-dimension framework with scoring guidelines

### Category-Specific Rubrics
1. **`rubric_agroecology.md`** — Ecological principles, biodiversity, system design
2. **`rubric_indigenous_and_traditional_knowledge.md`** — TEK, seed sovereignty, cultural practices
3. **`rubric_integrated_farming_systems.md`** — Crop-livestock integration, IPM, system optimization
4. **`rubric_natural_fertilizers.md`** — Compost, manures, biofertilizers, nutrient management
5. **`rubric_organic_farming.md`** — Certification standards, organic management
6. **`rubric_permaculture.md`** — Design principles, zones/sectors, guilds
7. **`rubric_regenerative_agriculture.md`** — Soil health, carbon sequestration, holistic management
8. **`rubric_soil_science_and_microbiology.md`** — Soil biology, microbial ecology, nutrient cycling
9. **`rubric_sustainable_water_management.md`** — Water conservation, irrigation, watershed management

## Quality Dimensions (1–5 Likert Scale)

Each rubric evaluates MCQs across 5 dimensions:

1. **Factual Correctness** — Scientific accuracy, current knowledge
2. **Question Clarity** — Unambiguous, expert-interpretable
3. **Distractor Quality** — Plausible to novices, clearly wrong to experts
4. **Domain Relevance** — Tests meaningful knowledge vs. trivia
5. **Difficulty Appropriateness** — Requires domain expertise, not common sense

**Composite Score** = mean of 5 dimensions

## Usage

### For LLM-as-Judge (Issue #54)

```python
# Load appropriate rubric for item's category
rubric_path = f"quality/rubrics/rubric_{item['category']}.md"
with open(rubric_path) as f:
    rubric = f.read()

# Pass rubric + item to LLM judge
# Request 5 dimension scores + justifications
# Calculate composite score
```

### For Human Expert Calibration (Issue #55)

1. Select 50–100 sample items per category
2. Have experts score using category rubric
3. Calculate inter-annotator agreement (Krippendorff's α)
4. Use as gold standard for validating LLM judge

### Acceptance Thresholds

Default recommendations (may vary by category):

- **Composite ≥ 4.0:** Accept (high quality)
- **3.5 ≤ Composite < 4.0:** Review for improvements
- **3.0 ≤ Composite < 3.5:** Flag for revision
- **Composite < 3.0:** Reject

**Critical dimension:** Factual Correctness should be ≥3 (reject scientifically incorrect items regardless of other scores).

## Category-Specific Anchors

Each category rubric includes:

- **Anchor examples** — Concrete examples of scores 5, 3, and 1 for each dimension
- **Red flags** — Common quality issues specific to that domain
- **Category-adjusted thresholds** — Domain-appropriate acceptance criteria

## Design Principles

1. **Expert-centric:** Rubrics define what domain experts value, not what's easy to automate
2. **Category-specific:** Each domain has unique quality considerations
3. **Educative distractors:** Wrong options should teach common misconceptions
4. **Practical relevance:** Prioritize applicable knowledge over trivia
5. **Cultural sensitivity:** Especially for indigenous knowledge category

## Next Steps

- **Issue #54:** Implement LLM-as-judge scoring with rubric lookup
- **Issue #55:** Conduct expert calibration and calculate IAA metrics
- **Iteration:** Refine rubrics based on expert feedback and judge performance

## References

- **AgXQA** (Kpodo et al., 2024): AgEES rubric for agricultural QA
- **GPQA** (Rein et al., 2024): Expert validation pipeline
- **Wang et al.** (Applied Sciences, 2025): Likert rubric for LLM-generated MCQs
- **RubricEval** (Bai et al., 2024): Rubric-based evaluation framework
