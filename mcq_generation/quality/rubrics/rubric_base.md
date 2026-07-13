# Base Quality Rubric Template

This document defines the 5-dimension quality assessment framework applied to all agricultural MCQ benchmark categories.

## Scoring Scale

All dimensions use a **1–5 Likert scale**:

- **5 = Excellent** — Meets all criteria at the highest standard
- **4 = Good** — Meets criteria with minor room for improvement
- **3 = Acceptable** — Meets minimum criteria but has noticeable limitations
- **2 = Poor** — Falls short of minimum criteria in important ways
- **1 = Unacceptable** — Fails to meet criteria; item should be rejected

## Quality Dimensions

### 1. Factual Correctness

**Definition:** Is the marked answer scientifically correct according to current domain knowledge and authoritative sources?

**Evaluation criteria:**
- The correct answer is unambiguously right based on established science
- No credible expert would dispute the correctness of the marked answer
- The question stem and options contain no factual errors
- If the question involves context-dependent answers, sufficient context is provided

**Red flags (score 1–2):**
- The marked answer is debatable or controversial
- Multiple answers could be considered correct
- Contains factual errors in stem or options
- Oversimplifies complex topics to the point of inaccuracy

### 2. Question Clarity

**Definition:** Would a domain expert interpret the question in only one way, understanding exactly what is being asked?

**Evaluation criteria:**
- Question stem is unambiguous and grammatically correct
- Technical terms are used precisely
- The scope of the question is clear (e.g., specific context, timeframe, conditions)
- No undefined pronouns or vague references

**Red flags (score 1–2):**
- Ambiguous wording that could lead to multiple interpretations
- Unclear what the question is asking
- Missing critical context
- Grammatical errors that obscure meaning

### 3. Distractor Quality

**Definition:** Are the incorrect options plausible to non-experts but definitively wrong to domain experts?

**Evaluation criteria:**
- Wrong options are tempting to someone with incomplete knowledge
- Distractors test common misconceptions or partial understanding
- All distractors are clearly incorrect to an expert
- Options are similar in length, structure, and specificity

**Red flags (score 1–2):**
- Obviously wrong options that anyone could eliminate
- Nonsensical or implausible distractors
- One option is much longer/shorter or more specific than others
- Options that are technically correct but less optimal (unless the question asks for "best")

### 4. Domain Relevance

**Definition:** Does the question test meaningful domain knowledge rather than peripheral trivia or tangential facts?

**Evaluation criteria:**
- Tests core concepts, principles, or practices in the domain
- Knowledge tested is applicable to real-world scenarios
- Connects to important domain competencies
- Not testing obscure facts unlikely to be known by practitioners

**Red flags (score 1–2):**
- Tests trivial details (e.g., specific page numbers, minor dates)
- Knowledge has no practical application
- Tests memorization of terminology without conceptual understanding
- Focuses on peripheral topics unrelated to domain competency

### 5. Difficulty Appropriateness

**Definition:** Does the question require genuine domain knowledge rather than common sense or general reasoning?

**Evaluation criteria:**
- Cannot be answered correctly by an educated non-expert using only reasoning
- Requires specific domain knowledge, experience, or training
- Appropriate difficulty for assessing domain competency
- Not so obscure that only specialists in a narrow sub-field would know

**Red flags (score 1–2):**
- Can be answered using common sense or general knowledge
- Obvious from context clues in the question itself
- Too easy (most non-experts would get it right)
- Too specialized (only researchers in a narrow sub-field would know)

## Usage Notes

### For LLM-as-Judge Scoring

When using these rubrics with LLM judges:

1. Provide the specific category rubric file (not just this base template)
2. Include 2–3 anchor examples per dimension showing score levels
3. Ask the judge to score each dimension independently
4. Request brief justification for each score
5. Calculate composite score as mean across dimensions

### For Human Expert Calibration

When calibrating human experts:

1. Have experts score 50–100 sample items using the category-specific rubric
2. Calculate inter-annotator agreement (Krippendorff's α)
3. Discuss disagreements to refine rubric anchors
4. Use expert scores as gold standard for validating LLM judge

### Composite Scoring

**Overall quality score** = mean of 5 dimension scores

**Acceptance thresholds** (recommended):
- **Composite ≥ 4.0** — High quality, keep
- **3.0 ≤ Composite < 4.0** — Acceptable, consider for revision
- **Composite < 3.0** — Reject or require major revision

Category-specific rubrics may adjust these thresholds based on domain standards.

## Related Documents

- Category-specific rubrics: `rubric_<category>.md`
- LLM judge implementation: `quality/llm_judge_scorer.py` (Issue #54)
- Judge calibration workflow: `quality/judge_calibration.py` (Issue #55)
