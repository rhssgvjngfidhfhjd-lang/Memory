#!/usr/bin/env bash
set -uo pipefail

BENCHMARK=${1:?usage: run_0910_graph_ablation_queue.sh memgallery|h2hmem|wma [suffix]}
RUN_SUFFIX=${2:-}
if [[ -n "$RUN_SUFFIX" && ! "$RUN_SUFFIX" =~ ^_[a-z0-9]+$ ]]; then
  echo "suffix must be empty or match _[a-z0-9]+: $RUN_SUFFIX" >&2
  exit 2
fi
WORKSPACE=/data/haozhen/Memory-clean
OFFLINE_ROOT="$WORKSPACE/Offline"
CONTROL_ROOT="$OFFLINE_ROOT/outputs/Ablation/_0910_graph_ablation_control"
PYTHON=/data/haozhen/miniconda3/envs/pipeline_repro/bin/python
JUDGE_KEY_FILE="$WORKSPACE/Nvida_api/Openrouter_api"
WANDB_PROJECT=hivemem-graph-ablation
WANDB_ENTITY=rhssgvjngfidhfhjd-nanyang-technological-university-singapore
EXPECTED_SOURCE_BUNDLE=2c89bfe6872a9c1113f2acbd95ea67e412ec9d3c9309da3ecbf3153d6cf9b774
EXPECTED_MANIFEST_SHA=590f187604245a7c2d446786662689afa8cdce855897e463694083fd0abb0d36
EXPECTED_EFFICIENCY_SHA=e86d8233ff0ae24a2edde996dd245c5f8fb527d617d82716f1eec07932ed8379
EXPECTED_DOC_SHA=78c75c990e1903aebdc878dd249c1d1ab0168ead0f1e79c63cce326a8949d3bf
EXPECTED_JUDGE_PROTOCOL_SHA=c83ddf6f54c2140af045a14f999713c664958bfe52e488b9c92ead952bf3a05c
EXPECTED_JUDGE_PROTOCOL_SNAPSHOT=evaluation_protocol_bundle@2026-09-09-wma-context-fix

case "$BENCHMARK" in
  memgallery)
    OUTPUT_BENCHMARK=MemGallery
    CONFIG="$OFFLINE_ROOT/configs/evidence_policy.json"
    EXPECTED_CONFIG_SHA=43321d9e9ceb9d12feb48a47549fa830b1f68bff72ac4e813a53dce3327e9773
    ENDPOINT=http://127.0.0.1:8013/v1
    CHECKPOINT="$OFFLINE_ROOT/outputs/PPO/MemGallery_0907PPO_a/checkpoints/epoch_005.pt"
    EXPECTED_CHECKPOINT_SHA=c766a9dfe9ac36fbd3b30da87533ccbf022f5398dffbf873272f88727dbc6618
    EXPECTED_TEST=275
    EXPECTED_PROMPT_SHA=ba624c662526600db61c3c33ce1c3ef5f3f62b3cc099fa57e7ef2bbc066391ff
    JUDGE_BENCHMARK=memgallery
    ;;
  h2hmem)
    OUTPUT_BENCHMARK=H2HMEM
    CONFIG="$OFFLINE_ROOT/configs/evidence_policy_h2hmem.json"
    EXPECTED_CONFIG_SHA=b9a800711235e08936cac5d8ba85381810abf149028b3441cd0e1c41cb562309
    ENDPOINT=http://127.0.0.1:8014/v1
    CHECKPOINT="$OFFLINE_ROOT/outputs/PPO/H2HMEM_0907PPO_b/checkpoints/epoch_005.pt"
    EXPECTED_CHECKPOINT_SHA=9c975a9909369ffdb01bbe24116b8898be543c0c66ac256261838044be5c57ad
    EXPECTED_TEST=360
    EXPECTED_PROMPT_SHA=4d50da58fedb3de7d185d5188f2815b4af5b2bc5c0c09e95627df481402a9f0f
    JUDGE_BENCHMARK=h2hmem
    ;;
  wma)
    OUTPUT_BENCHMARK=WorldMemArena
    CONFIG="$OFFLINE_ROOT/configs/evidence_policy_wma.json"
    EXPECTED_CONFIG_SHA=db8a88f7027218eefe4ca0d384b0855ba0b1fdfe32437152ca1ee6de479489ae
    ENDPOINT=http://127.0.0.1:8015/v1
    CHECKPOINT="$OFFLINE_ROOT/outputs/PPO/WMA_0909PPO_a/checkpoints/epoch_005.pt"
    EXPECTED_CHECKPOINT_SHA=14fbbf24aa388f0a4f06373cadb91baf02b4aaaaef260965ec0b9fd2dc7d7847
    EXPECTED_TEST=440
    EXPECTED_PROMPT_SHA=6b37fbcdf0922bd7ae049be6a2e024a4532de434fdc5ae41b229d5d73c113250
    JUDGE_BENCHMARK=worldmemarena
    ;;
  *)
    echo "unsupported benchmark: $BENCHMARK" >&2
    exit 2
    ;;
esac

export PYTHONPATH="$OFFLINE_ROOT/src:$OFFLINE_ROOT:$WORKSPACE"
export CUDA_VISIBLE_DEVICES=""
mkdir -p "$CONTROL_ROOT/logs"
QUEUE_LOG="$CONTROL_ROOT/logs/${BENCHMARK}_queue.log"
QUEUE_STATUS="$CONTROL_ROOT/${BENCHMARK}_queue.status"

timestamp() { date --iso-8601=seconds; }

set_queue_status() {
  printf '%s %s\n' "$1" "$(timestamp)" | tee "$QUEUE_STATUS"
}

set_run_status() {
  local output_dir=$1
  local stage=$2
  mkdir -p "$output_dir/run_control"
  printf '%s %s\n' "$stage" "$(timestamp)" | tee "$output_dir/run_control/status.txt"
}

run_logged() {
  local log_path=$1
  shift
  "$@" 2>&1 | tee -a "$log_path"
  return "${PIPESTATUS[0]}"
}

sha_of() { sha256sum "$1" | awk '{print $1}'; }

source_bundle_sha() {
  (
    cd "$WORKSPACE" || exit 1
    sha256sum \
      answer_prompts.py \
      Offline/src/benchmarks/h2hmem_harness/prompts.py \
      Offline/src/benchmarks/wma_harness/runner/prompts.py \
      Offline/src/benchmarks/memgallery_harness/runner/prompts.py \
      Offline/scripts/evidence_policy.py \
      Offline/src/evidence_policy/retrieval.py \
      Offline/src/evidence_policy/rollout.py \
      Offline/src/hive_mem/retriever.py \
      Offline/src/benchmarks/memgallery_harness/runner/metrics.py \
      Offline/scripts/judge_results_llm_parallel.py | sha256sum | awk '{print $1}'
  )
}

check_frozen_inputs() {
  [[ "$(source_bundle_sha)" == "$EXPECTED_SOURCE_BUNDLE" ]] || return 1
  [[ "$(sha_of "$CONFIG")" == "$EXPECTED_CONFIG_SHA" ]] || return 1
  [[ "$(sha_of "$CHECKPOINT")" == "$EXPECTED_CHECKPOINT_SHA" ]] || return 1
  [[ "$(sha_of "$OFFLINE_ROOT/configs/multimodal_split_manifest.json")" == "$EXPECTED_MANIFEST_SHA" ]] || return 1
  [[ "$(sha_of "$OFFLINE_ROOT/configs/model_efficiency.json")" == "$EXPECTED_EFFICIENCY_SHA" ]] || return 1
  [[ "$(sha_of "$OFFLINE_ROOT/docs/0910消融实验.md")" == "$EXPECTED_DOC_SHA" ]] || return 1
}

wait_for_endpoint() {
  while true; do
    model=$(curl -sf --max-time 10 "$ENDPOINT/models" 2>/dev/null | "$PYTHON" -c \
      'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)
    if [[ "$model" == "Qwen/Qwen3-VL-4B-Instruct" ]]; then
      return 0
    fi
    printf '%s endpoint_wait endpoint=%s model=%s\n' "$(timestamp)" "$ENDPOINT" "${model:-unavailable}" | tee -a "$QUEUE_LOG"
    sleep 60
  done
}

validate_test() {
  local output_dir=$1
  local run_id=$2
  local mode=$3
  local top_k=$4
  local base_top5_id=$5
  "$PYTHON" - "$output_dir" "$run_id" "$mode" "$top_k" "$base_top5_id" "$EXPECTED_TEST" "$EXPECTED_PROMPT_SHA" "$BENCHMARK" <<'PY'
import json, re, sys
from pathlib import Path
from evidence_policy.split_manifest import SplitManifestIndex

root=Path(sys.argv[1]); run_id=sys.argv[2]; mode=sys.argv[3]
top_k=int(sys.argv[4]); base_top5_id=sys.argv[5]; expected=int(sys.argv[6])
prompt_sha=sys.argv[7]; benchmark=sys.argv[8]
result=root/'eval'/'test_ppo'
metrics=json.loads((result/'metrics.json').read_text())
rows=[json.loads(line) for line in (result/'rollouts.jsonl').open() if line.strip()]
assert metrics['count']==expected==len(rows)
assert metrics['errors']==0 and metrics['cached_rollouts']==0
assert len({row['query_id'] for row in rows})==expected
assert {row.get('prompt_sha256') for row in rows}=={prompt_sha}
assert all(re.fullmatch(r'\s*<answer>[^<]+</answer>\s*', row['answer_raw_response'], re.S) for row in rows)

manifest=SplitManifestIndex('/data/haozhen/Memory-clean/Offline/configs/multimodal_split_manifest.json')
sources={
 'memgallery':('mem_gallery',),
 'h2hmem':('h2hmem_dyadic','h2hmem_multiparty'),
 'wma':('worldmemarena_lifelong',),
}[benchmark]
expected_ids=set().union(*(set(manifest.question_ids('test',data_source=source)) for source in sources))
actual_ids={row['manifest_question_id'] for row in rows}
assert actual_ids==expected_ids, (len(actual_ids),len(expected_ids))

for row in rows:
    hits=row['retrieval_top_k']; ids=row['retrieval_final_ids']
    assert row['retrieval_mode']==mode
    assert row['vector_k']==top_k
    assert ids==[hit['memory_id'] for hit in hits]
    assert len(ids)==len(set(ids))
    if mode=='vector':
        assert len(ids)==top_k and all(hit['via']=='vector' for hit in hits)
    elif mode=='random_append':
        assert len(ids)==7 and row['append_k_actual']==2
        assert [hit['via'] for hit in hits]==['vector']*5+['random']*2
        assert row['retrieval_seed'] is not None
    else:
        assert 5 <= len(ids) <= 7
        assert all(hit['via']=='vector' for hit in hits[:5])
        assert all(hit['via']=='graph' for hit in hits[5:])
    if benchmark=='wma':
        visible=set(row['visible_sessions'])
        assert all(hit['session_id'] in visible for hit in hits)

if benchmark=='wma':
    assert sum(row['category']=='MB' for row in rows)==40

base_run=root.parent/base_top5_id/'eval'/'test_ppo'/'rollouts.jsonl'
if run_id!=base_top5_id:
    base={row['manifest_question_id']:row for row in (json.loads(line) for line in base_run.open() if line.strip())}
    for row in rows:
        reference=base[row['manifest_question_id']]
        assert row['retrieval_final_ids'][:5]==reference['retrieval_final_ids']
        assert [a['mask'] for a in row['actions'][:5]]==[a['mask'] for a in reference['actions']]

for name in ('results.json','retrieval_trace.jsonl','call_trace.jsonl','call_metrics.json','efficiency_metrics.json','summary.json'):
    assert (result/name).is_file(), name
eff=json.loads((result/'efficiency_metrics.json').read_text())
for section in ('cost_mb','cost_qa','latency_mb','latency_qa'):
    assert eff[section]['available'] is True, (section,eff[section].get('reason'))
assert eff['latency_qa']['denominator_unit']=='QA'
assert eff['latency_qa']['num_samples']==expected
PY
}

validate_judge() {
  local result=$1
  "$PYTHON" - "$result" "$EXPECTED_TEST" "$EXPECTED_JUDGE_PROTOCOL_SHA" "$EXPECTED_JUDGE_PROTOCOL_SNAPSHOT" <<'PY'
import json,sys
from pathlib import Path
result=Path(sys.argv[1]); expected=int(sys.argv[2]); protocol_sha=sys.argv[3]; snapshot=sys.argv[4]
m=json.loads((result/'llm_judge_metrics.json').read_text())
assert m['count']==m['valid_count']==expected
assert m['judge_errors']==0 and m['coverage']==m['completion']==1.0
assert m['protocol_snapshot']==snapshot
c=json.loads((result/'llm_judge_checkpoint.json').read_text())
assert c['signature']['judge']['protocol_sha256']==protocol_sha
s=json.loads((result/'summary.json').read_text())
assert s['llm_judge']==m['accuracy']
PY
}

write_run_manifest() {
  local output_dir=$1 run_id=$2 mode=$3 top_k=$4 seed=$5
  "$PYTHON" - "$output_dir" "$run_id" "$BENCHMARK" "$mode" "$top_k" "$seed" \
    "$CONFIG" "$CHECKPOINT" "$ENDPOINT" "$EXPECTED_SOURCE_BUNDLE" "$EXPECTED_DOC_SHA" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
out=Path(sys.argv[1]); run_id=sys.argv[2]; benchmark=sys.argv[3]; mode=sys.argv[4]
top_k=int(sys.argv[5]); seed=None if sys.argv[6]=='none' else int(sys.argv[6])
config=Path(sys.argv[7]); checkpoint=Path(sys.argv[8])
rows=[json.loads(line) for line in (out/'eval'/'test_ppo'/'rollouts.jsonl').open() if line.strip()]
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
payload={
 'schema_version':1,'experiment':'0910_graph_ablation','run_id':run_id,
 'benchmark':benchmark,'split':'test','question_count':len(rows),
 'retrieval_mode':mode,'vector_k':top_k,'random_seed':seed,
 'config_path':str(config),'config_sha256':sha(config),
 'checkpoint_path':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
 'model_endpoint':sys.argv[9],'source_bundle_sha256':sys.argv[10],
 'protocol_doc':'/data/haozhen/Memory-clean/Offline/docs/0910消融实验.md',
 'protocol_doc_sha256':sys.argv[11],
 'prompt_version':rows[0]['prompt_version'],'prompt_source':rows[0]['prompt_source'],
 'prompt_sha256':rows[0]['prompt_sha256'],
 'retrieval_signatures':sorted({row['retrieval_signature'] for row in rows}),
 'completed_at':datetime.now(timezone.utc).astimezone().isoformat(),
}
(out/'run_manifest.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
PY
}

upload_wandb_summary() {
  local output_dir=$1 run_name=$2
  "$PYTHON" - "$output_dir" "$run_name" "$WANDB_PROJECT" "$WANDB_ENTITY" <<'PY'
import json,secrets,sys
from pathlib import Path
import wandb
root=Path(sys.argv[1]); name=sys.argv[2]
metrics=json.loads((root/'eval'/'test_ppo'/'summary.json').read_text())
config=json.loads((root/'config.json').read_text())
run_id=secrets.token_hex(4)
run=wandb.init(project=sys.argv[3],entity=sys.argv[4],name=name,id=run_id,resume='never',job_type='ablation-test',config=config,tags=['hivemem','graph-ablation','0910'])
summary={'test/f1':metrics['f1'],'test/em':metrics['em'],'test/judge':metrics['llm_judge'],'test/count':metrics['count']}
for section in ('cost_mb','cost_qa','latency_mb','latency_qa'):
    values=metrics[section]
    key='mean_per_sample_usd' if section.startswith('cost') else 'mean_per_sample_seconds'
    summary[f'test/{section}']=values[key]
summary['test/calls_mb_plus_qa']=metrics['calls']['total']['mean_per_sample']
run.log(summary)
for key,value in summary.items(): run.summary[key]=value
url=str(run.url or '')
run.finish()
control=root/'run_control'; control.mkdir(parents=True,exist_ok=True)
(control/'wandb.json').write_text(json.dumps({'project':sys.argv[3],'entity':sys.argv[4],'name':name,'run_id':run_id,'status':'finished','url':url},indent=2)+'\n')
print(url)
PY
}

set_queue_status running
BASE_TOP5_ID="0910_graph_ablation_top5${RUN_SUFFIX}"
for spec in \
  '0910_graph_ablation_top5:vector:5:none' \
  '0910_graph_ablation_top7:vector:7:none' \
  '0910_graph_ablation_random2_seed42:random_append:5:42' \
  '0910_graph_ablation_random2_seed43:random_append:5:43' \
  '0910_graph_ablation_random2_seed44:random_append:5:44' \
  '0910_graph_ablation_graph2:graph_append:5:none'; do
  IFS=: read -r base_run_id mode top_k seed <<< "$spec"
  run_id="${base_run_id}${RUN_SUFFIX}"
  output_dir="$OFFLINE_ROOT/outputs/$OUTPUT_BENCHMARK/HiveMem/$run_id"
  result_dir="$output_dir/eval/test_ppo"

  if [[ -e "$output_dir" ]]; then
    printf '%s output_collision run=%s path=%s\n' "$(timestamp)" "$run_id" "$output_dir" | tee -a "$QUEUE_LOG"
    set_queue_status failed
    exit 1
  fi
  if ! check_frozen_inputs; then
    printf '%s frozen_input_changed run=%s\n' "$(timestamp)" "$run_id" | tee -a "$QUEUE_LOG"
    set_queue_status failed
    exit 1
  fi
  wait_for_endpoint
  mkdir -p "$output_dir/run_control"
  printf '%s\n' "$(timestamp)" > "$output_dir/run_control/started_at.txt"
  set_run_status "$output_dir" qa_running
  printf '%s run_start run=%s mode=%s top_k=%s seed=%s\n' "$(timestamp)" "$run_id" "$mode" "$top_k" "$seed" | tee -a "$QUEUE_LOG"

  seed_args=()
  if [[ "$seed" != none ]]; then seed_args=(--retrieval-seed "$seed"); fi
  if ! run_logged "$output_dir/qa.log" \
    "$PYTHON" "$OFFLINE_ROOT/scripts/evidence_policy.py" \
      --config "$CONFIG" --output-dir "$output_dir" --model-base-url "$ENDPOINT" \
      --retrieval-mode "$mode" --top-k "$top_k" "${seed_args[@]}" \
      eval --strategy ppo --split test --checkpoint "$CHECKPOINT" --device cpu; then
    set_run_status "$output_dir" qa_failed
    set_queue_status failed
    exit 1
  fi
  if ! validate_test "$output_dir" "$run_id" "$mode" "$top_k" "$BASE_TOP5_ID"; then
    set_run_status "$output_dir" qa_acceptance_failed
    set_queue_status failed
    exit 1
  fi
  write_run_manifest "$output_dir" "$run_id" "$mode" "$top_k" "$seed"

  set_run_status "$output_dir" judge_running
  if ! run_logged "$result_dir/llm_judge.log" \
    "$PYTHON" "$OFFLINE_ROOT/scripts/judge_results_llm_parallel.py" \
      --benchmark "$JUDGE_BENCHMARK" --results "$result_dir/rollouts.jsonl" \
      --out-dir "$result_dir" --key-file "$JUDGE_KEY_FILE" \
      --model openai/gpt-4o-mini --workers 32 --timeout 60 --retries 2 \
      --max-tokens 512 --checkpoint-every 25 --resume; then
    set_run_status "$output_dir" judge_failed
    set_queue_status failed
    exit 1
  fi
  if ! validate_judge "$result_dir"; then
    set_run_status "$output_dir" judge_acceptance_failed
    set_queue_status failed
    exit 1
  fi

  if ! upload_wandb_summary "$output_dir" "$OUTPUT_BENCHMARK-$run_id" >> "$output_dir/wandb_upload.log" 2>&1; then
    printf '%s wandb_warning run=%s local_results_authoritative\n' "$(timestamp)" "$run_id" | tee -a "$QUEUE_LOG"
  fi
  set_run_status "$output_dir" complete
  printf '%s run_complete run=%s\n' "$(timestamp)" "$run_id" | tee -a "$QUEUE_LOG"
done
set_queue_status complete
