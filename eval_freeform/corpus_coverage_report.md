# Free-form eval corpus eligibility coverage

- Screened directories: 1854 / 1854 observed top-level corpus directories.
- Coverage vs observed corpus: 100.00%.
- Coverage vs issue-stated 1,855-doc corpus: 1854 / 1855 = 99.95%.
- Markdown files found: 1586.
- Eligible documents: 1013.
- Ineligible documents: 841.
- Conclusion: most of the screened corpus is on-topic.
- Partition seed: 20260715.

The observed corpus has 1,854 top-level directories, not the 1,855 stated in #42; 268 of those directories contain no markdown file and were still counted as screened/ineligible.
Because every observed top-level directory was screened, no early stopping justification is needed.

## Topic counts

- agroecology_systems: 34
- crop_horticulture: 743
- pest_disease: 43
- soil_fertility: 140
- water_irrigation: 53

## Slice sizes

- multi_hop_factual: 203 eligible source docs
- diagnostic_reasoning: 203 eligible source docs
- free_form_explanation: 203 eligible source docs
- constraint_satisfaction: 202 eligible source docs
- quantitative: 202 eligible source docs

## Heuristic

Each directory was screened using its title and opening prose after markdown cleanup. A document was eligible when it had at least 250 prose-like words and practical farming, gardening, horticulture, soil, water, pest, crop, or agroecology signal. Directories with missing markdown, thin extracted text, off-topic scientific/medical material, reference-only front matter, or duplicate opening prose were marked ineligible.
