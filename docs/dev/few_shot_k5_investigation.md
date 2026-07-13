# Few-Shot k5 Investigation

Evidence drawn from two consecutive smoke runs on 5 validation cases each:

- **Run 1** (`7940526e`, 2026-06-26 01:24): before the suffix fix
- **Run 2** (`98e69d3c`, 2026-06-26 10:37): after adding `_EXAMPLES_SUFFIX`

The validation cases in both runs are `contractnli_validation_000020` through `_000024`.

---

## Token Count Measurements

### System prompt token counts (both runs use the same prompts in Run 2)

| Strategy | System prompt chars | Approx tokens |
|---|---|---|
| baseline_scaffold | 1,404 | 351 |
| baseline_few_shot_k3 | 3,432 | 858 |
| baseline_few_shot_k5 | 3,987 | 996 |

`count_tokens_approx` uses `len(text) // 4`, the same function used by the budget guard.

### Per-case total (system + user) for Run 2 smoke cases

The user message is `json.dumps(case.input)`. ContractNLI inputs range from short single-sentence clauses to long multi-paragraph extracts.

| Case ID | User msg tokens | k3 total | k5 total |
|---|---|---|---|
| contractnli_validation_000020 | 158 | 1,016 | 1,154 |
| contractnli_validation_000021 | 118 | 976 | 1,114 |
| contractnli_validation_000022 | 123 | 981 | 1,119 |
| contractnli_validation_000023 | 530 | 1,388 | 1,526 |
| contractnli_validation_000024 | 88 | 946 | 1,084 |
| **Mean** | **203** | **1,061** | **1,199** |
| **Delta (k5 − k3)** | — | — | **+138** |

**Model context window**: gemma-4-E4B-it-Q4_K_M.gguf is served locally via llama.cpp. Gemma 4 natively supports ≥ 8,192 tokens; the Q4_K_M quantisation at 4B parameters typically runs with ≥ 8 K context on the hardware in use. The `max_prompt_tokens` budget guard in the task YAML is set to 10,000.

**k5 at mean consumes ≈ 12% of the 10,000-token budget** and well under 20% of any realistic model context window. Context length is not the cause of the failure.

---

## Prompt Comparison (case 000021, k3 vs k5, Run 2)

Case 000021 expected label: **entailment**. Both k3 and k5 get it wrong in Run 2.

### k3 prompt (abridged — full prompt is 3,432 chars)

```
# Role
You are a legal natural language inference assistant. …

# Task
Classify the logical relationship …

# Input
Contract clause:
{{ premise }}
…

# Output Format
Return only valid JSON matching this schema:
```json
{
  "label": "<one allowed value>"
}
```

# Allowed Values
- contradiction
- entailment
- neutral

# Rules
- …

# Input Boundary
- …

# Few-Shot Examples

Use these examples as guidance.

### Example 1
Input:
```json
{
  "hypothesis": "Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.",
  "premise": "5. All Confidential Information in any form …"
}
```
Output:
```json
{"label": "entailment"}
```

### Example 2
Input:
```json
{
  "hypothesis": "Receiving Party shall not reverse engineer any objects …",
  "premise": "2.3 Provided that the Recipient has a written agreement …"
}
```
Output:
```json
{"label": "neutral"}
```

### Example 3
Input:
```json
{
  "hypothesis": "Agreement shall not grant Receiving Party any right to Confidential Information.",
  "premise": "4. Nothing in this Agreement is to be construed as granting …"
}
```
Output:
```json
{"label": "entailment"}
```

---

Classify the input you receive using the pattern above.
```

**User message sent to model:**
```json
{"hypothesis": "Receiving Party may independently develop information similar to Confidential Information.", "premise": "Confidential Information does not include information: (i) which is or becomes public knowledge …"}
```

**Raw student response (k3, case 000021):**
```
{"premise": "Confidential Information does not include information: (i) which is or becomes public knowledge and public property in any way without breach of this Agreement by the Recipient; (ii) which the Recipient can show has been known or has been developed by or for the Recipient at any time independently of the information disclosed to it by the Discloser;", "hypothesis": "Receiving Party may independently develop information similar to Confidential Information."}
```

The model output the **entire test-case input** as a `{"premise": "…", "hypothesis": "…"}` dict, not a `{"label": "…"}` classification. `actual_output.get("label")` returns `""`.

---

### k5 prompt (abridged — full prompt is 3,987 chars)

Identical to k3 through Example 3, then adds:

```
### Example 4
Input:
```json
{
  "hypothesis": "Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.",
  "premise": "11. The Recipient agrees to keep the contents of this Agreement and any related negotiations …"
}
```
Output:
```json
{"label": "entailment"}
```

---

Classify the input you receive using the pattern above.
```

**User message**: same as k3.

**Raw student response (k5, case 000021):**
```
{"premise": "Confidential Information does not include information: (i) which is or becomes public knowledge and public property in any way without breach of this Agreement by the Recipient; (ii) which the Recipient can show has been known or has been developed by or for the Recipient at any time independently of the information disclosed to it by the Discloser;", "hypothesis": "Receiving Party may independently develop information similar to Confidential Information."}
```

Identical failure: the model echoed the test-case input as a premise/hypothesis dict instead of classifying it.

---

## Example Selection Analysis

### Selection method

`select_deterministic_few_shot_examples` in `baseline_strategies.py:47` performs label-balanced **round-robin** selection:

1. Sort training cases by `case.id` (alphabetical/lexicographic).
2. Group by label.
3. Cycle through sorted labels (alphabetically: `contradiction`, `entailment`, `neutral`), picking the next case for each label in turn until `k` examples are collected.

This is **not** similarity-based. Examples are not chosen by proximity to the test case. Selection is fully deterministic given the same training pool.

### Examples selected (Run 2 smoke run)

The smoke run training pool had no labeled `contradiction` cases visible to the baseline selector (the validation split was used for eval; training examples came from the configured training split, and in this 5-case smoke run the available labeled examples happened to yield none tagged `contradiction`).

| Slot | k3 | k5 |
|---|---|---|
| 1 | entailment — "Receiving Party shall destroy or return…" | same |
| 2 | neutral — "Receiving Party shall not reverse engineer…" | same |
| 3 | entailment — "Agreement shall not grant…" | same |
| 4 | — | entailment — "Receiving Party shall not disclose the fact…" |

**k5 examples are a superset of k3 examples.** Example 4 is added; there is no contradiction example in either set. The budget guard (`max_prompt_tokens = 10,000`) did not trim; all four examples fit comfortably (996 tokens vs 10,000 limit). The fifth candidate was unavailable in the training pool for this run.

Selection is **deterministic** (no random seed involved). Re-running on the same training split would produce the same examples.

No example shares its label with the test case in a way that distinguishes k3 from k5; both sets are weighted toward `entailment`.

---

## Response Quality Analysis

Full per-case breakdown for Run 2 across all three strategies:

| Case | Expected | scaffold | k3 output | k5 output |
|---|---|---|---|---|
| 000020 | entailment | entailment ✓ | `{` (dict, wrong label) | `"Some obligations of Agreement may survive termination of Agreement."` |
| 000021 | entailment | entailment ✓ | `{"premise": "…", "hypothesis": "…"}` (echoed input) | `{"premise": "…", "hypothesis": "…"}` (echoed input) |
| 000022 | neutral | entailment ✗ | `"Receiving Party may retain one (1) copy…"` (premise text) | `"entailment"` (wrong label) |
| 000023 | entailment | entailment ✓ | `"entailment"` ✓ | `"entailment"` ✓ |
| 000024 | neutral | neutral ✓ | `"neutral"` ✓ | `{` (dict, wrong label) |

Grammar-constrained generation (llama.cpp `response_format=json_schema`, native mode) forces all output into the `_StudentOutputSchema` wrapper `{"raw": …}`. The grammar constraint means the response is always valid JSON and `malformed=False`. The `raw` field carries the model's actual intent:

- When the model correctly classifies, `raw = "entailment"` (string) or `raw = {"label": "entailment"}` (dict).
- When the model continues the pattern, `raw = {"premise": "…", "hypothesis": "…"}` (the test-case input re-echoed) or `raw = "Some obligations…"` (a hypothesis sentence from training data).

**None of the responses are truncated mid-output.** The JSON is always complete. The label field is either absent (wrong dict shape) or contains the wrong value. No evidence of mid-sentence cuts or context overflow.

---

## Run 1 vs Run 2 comparison (what the "fix" changed)

The only change between Run 1 and Run 2 was the addition of `_EXAMPLES_SUFFIX`:

```diff
+
+---
+
+Classify the input you receive using the pattern above.
```

| Run | k3 F1 | k5 F1 | What went wrong |
|---|---|---|---|
| Run 1 (no suffix) | 0.1111 | 0.2500 | Both strategies echo premise text or input JSON for most cases. k5 happened to output `"entailment"` for cases 000020–000021–000023, all of which were entailment; that accidental hit pushed its F1 higher. |
| Run 2 (with suffix) | 0.2333 | 0.0800 | The suffix partially breaks the label-echoing in k3 (2/5 correct vs 1/5). For k5, the stronger 4-example momentum produces worse alternatives (hypothesis text, echoed input JSON) instead of the label. The accidental entailment bias from Run 1 is gone. |

k5 outperforming k3 in Run 1 was a coincidence: the model was biased toward outputting `"entailment"` (the label of the final few-shot example), and 3 of the 5 test cases were in fact `entailment`. The suffix disrupted that bias without providing a reliable alternative for k5.

---

## Example Format Analysis

```
### Example N
Input:
```json
{
  "hypothesis": "…",    ← sorted keys, 2-space indent
  "premise": "…"
}
```
Output:
```json
{"label": "contradiction"}    ← compact, no indent
```
```

**Separator between examples**: blank line between each `### Example N` block.

**Separator between examples and test case**: `\n\n---\n\nClassify the input you receive using the pattern above.` (single horizontal rule + one instruction line).

**Instruction placement**: output format instruction (`# Output Format`, `# Allowed Values`) appears **before** the examples, meaning the model sees the instruction, then overwrites that context with 3–4 demonstration turns. By the time it sees the suffix, the instruction is far in the past.

**User message format vs example Input format**:

| Where | Format |
|---|---|
| Example Input | `json.dumps(case.input, sort_keys=True, indent=2)` → pretty-printed, sorted keys |
| User message (test case) | `json.dumps(case.input)` → compact, insertion-order keys |

Both are JSON objects with the same two keys (`"premise"`, `"hypothesis"`). Despite the minor whitespace/key-order difference, the model perceives them as structurally equivalent: it is receiving another JSON object of the same shape as every example Input it just processed. The `---` separator and the "Classify the input" line are insufficient to override this strong structural cue.

**Secondary issue — YAML parsing bug in `task.yaml`**: Unquoted rule strings containing `: ` (colon-space) are parsed by YAML as mappings:
```yaml
task_scope:
  - Classify the relationship into exactly one of: entailment, contradiction, or neutral.
```
YAML parses this list item as `{"Classify the relationship into exactly one of": "entailment, contradiction, or neutral."}`. The Rules section in the compiled prompt reads:
```
- {'Classify the relationship into exactly one of': 'entailment, contradiction, or neutral.'}
```
This Python-dict repr string is syntactically confusing but affects k3 and k5 identically, so it is not the cause of the k3/k5 differential.

---

## Root Cause Hypothesis

Ranked by confidence.

### 1. User message format is structurally identical to example Input (very high confidence)

The user message `json.dumps(case.input)` produces a JSON object with the same two keys as every example's `Input:` block. After seeing 3–4 `Input: {…} / Output: {…}` turns, the model interprets the new JSON object as "Example N+1 Input" and generates what it expects to come next: either another `Input:` body (hypothesis text, premise text, or the same JSON) or a partial `Output:` dict.

With k3 (3 examples) the pattern pressure is moderate; the suffix often redirects the model. With k5 (4 examples) the pattern is reinforced more deeply, making the suffix insufficient and causing more consistent mis-completion.

### 2. Accidental advantage explains Run 1 k5 > k3 (very high confidence)

In Run 1 (no suffix), the model in the k5 condition was heavily biased toward continuing the pattern by copying the last example's output label (`"entailment"`). This happened to match 3/5 test cases. It was not evidence of better understanding; the suffix removed this bias without replacing it with correct classification.

### 3. Token budget / context window pressure (very low confidence)

Ruled out. k5 uses ≈ 12% of the 10,000-token budget, and no more than 20% of any reasonable llama.cpp context window for a 4B model. There is no truncation or overflow evidence in the responses.

---

## Recommended Fix

### Primary fix: render the user message using the task input template

Change `_call_student` in `evaluator.py:111`:

```python
# current
user_prompt = json.dumps(case.input)

# proposed
user_prompt = _render_input_from_template(task.input_template, case.input)
```

Where `_render_input_from_template` substitutes `case.input` fields into `task.input_template` (replacing `{{ premise }}` → actual clause text, `{{ hypothesis }}` → actual statement text).

The resulting user message for ContractNLI would be:
```
Contract clause:
[actual clause text]

Legal statement:
[actual hypothesis text]
```

This is structurally distinct from the few-shot example Input format (indented JSON fenced block). The model can no longer mistake the test case for "another example to continue."

**Tradeoff**: This change affects all strategies uniformly (including baseline_scaffold, retrieval_few_shot, and the distilled prompt strategies). Each task would need a functioning `input_template`; tasks without templates would need a fallback. The evaluator currently does not receive a `task` argument — `evaluate_prompt` signature would need to be extended, or `user_prompt` pre-rendered before the call.

### Secondary fix: quote YAML rule strings in task.yaml

Wrap task_scope items that contain `: ` in YAML quotes to prevent mapping parse:

```yaml
task_scope:
  - "Classify the relationship into exactly one of: entailment, contradiction, or neutral."
  - "The label must exactly match one of: entailment, contradiction, neutral."
```

This fixes the garbled Rules section (`{'Classify the relationship into exactly one of': …}`) in all baseline prompts. Not the cause of the k3/k5 gap but a correctness bug in its own right.

### Alternative (lower confidence): stronger boundary in the prompt

Replace the current suffix with a more forceful instruction section:

```markdown
---

# Input to Classify

The following is the actual input to classify. It is NOT another example.
Respond only with the required JSON output.
```

This is easier to implement (no evaluator changes) but relies on instruction-following, which the model demonstrated it does only partially in Run 2.
