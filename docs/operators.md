# Perturbation operators

One operator per pair, applied to a copy of one file with a per-permutation
derived seed, deterministic across re-runs (the controlled-variable
invariant). `traxr operators` prints this catalog from the installed
package.

## Tabular (CSV / XLSX): any agent

Delivered by file round-trip: parse → perturb → re-serialize to disk.

| operator | what it does |
|---|---|
| `column_swap` | swaps the values of two columns |
| `label_corrupt` | corrupts categorical labels |
| `data_type_corrupt` | breaks cell types (numbers → text, …) |
| `row_duplicate` | duplicates rows |
| `irrelevant_columns` | injects plausible distractor columns |
| `unit_change` | rescales numeric columns as if units changed |
| `null_content` | replaces the file with empty content |

## Text (TXT / MD): any agent

Also file round-trip.

| operator | what it does |
|---|---|
| `ocr_noise` | character-level OCR-style corruption |
| `number_corruption` | changes numeric values in running text |
| `text_redaction` | blacks out spans |
| `paragraph_shuffle` | reorders paragraphs |
| `encoding_error` | mojibake-style encoding damage |
| `section_removal` | deletes a section |
| `null_content` | empty file |

## PDF: any agent

Delivered by **surgical in-place edits** (PyMuPDF): the perturbed PDF on
disk differs only at the edit loci, preserving extraction fidelity
everywhere else. Visual fidelity is *not* guaranteed at edit spans;
agents doing layout/visual analysis may notice the reinserted font.

| operator | what it does |
|---|---|
| `number_corruption` | edits numbers in place |
| `text_redaction` | redacts spans in place |
| `section_removal` | removes a contiguous block |
| `page_removal` | drops a page |
| `page_shuffle` | reorders pages |
| `null_content` | blank document |

## PDF: built-in agent only

Whole-text-flow operators (`ocr_noise`, `paragraph_shuffle`,
`encoding_error`) cannot be applied surgically to a PDF. For the built-in
reference agent they are delivered by **content injection**: the perturbed
extracted text is handed to the agent's PDF tool instead of the file being
modified.
