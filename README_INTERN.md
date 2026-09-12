# Unified Final-Evaluation Protocol

This bundle is the source of truth for evaluating external baselines with the
same protocol as the main project. Keep the files under `agentic_memrl/`
unchanged. Adapt only the baseline-specific result exporter.

## Protocol summary

- Judge: OpenRouter `openai/gpt-4o-mini`
- Temperature: `0.0`
- Maximum judge output: `512` tokens
- Judge inputs: prediction and ground-truth references only
- MemEye: open-answer subset only; official judge quantization is
  `<0.25 -> 0`, `[0.25,0.75) -> 0.5`, `>=0.75 -> 1`
- F1/EM/judge: query-wise mean
- QA cost and external-model calls: sum over QA queries divided by the number
  of conversations
- QA latency: query-wise mean
- Report results separately for every dataset

## Prompt to give Codex

```text
I attached the unified final-evaluation protocol used by our main method.
Read every attached file completely before making changes, then inspect this
baseline repository and adapt its saved predictions to the protocol.

Requirements:
1. Treat judge_protocols.py as the only source of truth for judge prompts,
   benchmark dispatch, labels, and score spaces. Call render_judge_prompt();
   do not copy, rewrite, simplify, or embellish its prompt strings.
2. Treat run_llm_judge.py as the only source of truth for judge requests,
   parsing, JSON repair, official score conversion, and resume caching.
3. Use OpenRouter openai/gpt-4o-mini with temperature=0.0 and
   max_new_tokens=512.
4. The only sample-dependent judge inputs are prediction and ground-truth
   references. Do not add the question, benchmark name, context, memory
   points, or retrieved evidence.
5. Evaluate only the MemEye open subset. Preserve its official score
   quantization: score<0.25 -> 0; 0.25<=score<0.75 -> 0.5; score>=0.75 -> 1.
6. Use local_metrics.py for F1/EM dispatch. Mem-Gallery, WorldMemArena, and
   H2HMem use the shared F1; MemEye and MemLens use their benchmark-specific
   official normalization/F1 implementations.
7. Report metrics per dataset. F1, EM, and judge score are query-wise means.
   QA cost and QA external-LLM calls are sums over queries divided by the
   number of conversations. QA latency is a query-wise mean.
8. calls_qa counts only external LLM/VLM calls made while answering QA. It
   excludes policy generation, retrieval-only calls, and all judge calls.
9. If this baseline cannot supply a resource metric reliably, mark it
   unavailable. Never invent a value or silently substitute zero.
10. Fail on missing dataset_source, conversation_id, prediction, references,
    or required MemLens question-type metadata. Do not add fallback behavior.
11. Modify only a baseline-specific adapter that exports the normalized input
    expected by the evaluator. Do not modify the attached protocol files.
12. Add conformance tests covering protocol dispatch, MemEye quantization,
    benchmark F1 normalization, and dataset-wise aggregation.

Before implementation, briefly report the baseline's current output schema,
the adapter you will add, and which cost/latency/call fields are measurable.
Then implement and test the adapter.
```

## Files

- `evaluation/judge_protocols.py`: prompts and benchmark dispatch
- `evaluation/run_llm_judge.py`: API call, parser, quantization, and cache
- `evaluation/local_metrics.py`: F1/EM dispatch
- `evaluation/finalize_artifacts.py`: normalized schema and aggregation
- `evaluation/evaluate_saved_responses.py`: end-to-end post-hoc evaluator
- `evaluation/JUDGE_PROTOCOLS.md`: human-readable protocol record
- `rewards/*.py`: exact normalization/F1 implementations and dependencies
- `runtime_metrics.py`: cost, latency, and external-call accounting
- `tests/test_final_evaluation.py`: protocol conformance tests

`finalize_artifacts.py` understands the main project's Verl dump. A baseline
with a different output format should add an adapter into the normalized
`responses.jsonl` schema rather than changing the evaluator.
