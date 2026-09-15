# Datasheet for MEMLENS

This datasheet follows the *Datasheets for Datasets* template by Gebru et al. (2021). It complements the machine-readable Croissant metadata at `metadata/croissant.json` and the human-readable description in the paper.

---

## 1. Motivation

**For what purpose was the dataset created?**
MEMLENS was created to evaluate long-horizon conversational memory in vision-language models. Existing memory benchmarks either restrict themselves to text-only conversations (e.g., LongMemEval, MemoryAgentBench) or to static multimodal documents (e.g., MMLongBench), and no prior protocol jointly compares long-context vision-language models (LC-VLMs) and memory-augmented agents at matched context lengths. MEMLENS fills that gap by providing a length-controlled, image-essential, multi-session benchmark that places both families under a unified evaluation protocol.

**Who created the dataset and on behalf of which entity?**
The dataset was created by a research team at the Hong Kong University of Science and Technology (HKUST), the Chinese University of Hong Kong (CUHK), NVIDIA AI Technology Center (NVAITC), and OmniMemory, led by Xiyu Ren and Zhaowei Wang under the supervision of Prof. Yangqiu Song.

**Who funded the creation of the dataset?**
This work was supported by NVIDIA AI Technology Center (NVAITC) and the Hong Kong University of Science and Technology.

**Any other comments?**
MEMLENS is the first benchmark to (a) ground ≥80% of its evaluation questions in image content by construction, (b) embed evidence in multi-session dialogues rather than static documents, and (c) instantiate the same 789 questions at four standardized context lengths (32K/64K/128K/256K tokens) for direct cross-length comparison.

---

## 2. Composition

**What do the instances that comprise the dataset represent?**
Each instance is a *question record* with: a stable string `question_id`; a `question_type` (one of IE, MSR, TR, KU, AR); a `context_length` (one of 32768/65536/131072/262144); the natural-language `question`; the `reference_answer`; a serialized list of `haystack_sessions` (multi-turn dialogues); per-session `haystack_dates`; the `evidence_session_ids` carrying the answer-bearing turns; and `image_paths` referencing the `images/` FileSet for every `<image>` token in the conversation.

**How many instances are there in total?**
789 questions, instantiated at four context lengths (3,156 question-length pairs total). All four context-length variants share the same 789 `question_id`s; only the surrounding haystack varies. A separate *indexing file* (`agent_subset_195.json`, no QA payload) lists a fixed stratified 195-question subset (61 IE / 35 MSR / 48 TR / 29 KU / 22 AR) on which all seven memory-augmented agents are evaluated; this subset is identified for reproducibility, not because it carries any data of its own.

**Does the dataset contain all possible instances or is it a sample of instances from a larger set?**
It is a curated sample. Construction begins with a topic ontology authored by the dataset team; for each topic the iCrawler image-search results are filtered through a watermark / stock-photo logo / copyright-overlay rejection classifier and CLIP/SigLIP relevance thresholds. Surviving image-topic pairs seed two-agent dialogue generation. Question generation, deterministic-rule normalization, judge-rubric tractability checks, and a 484-question human consensus audit prune the final set down to 789 release-quality questions.

**What data does each instance consist of?**
Plain-text question, answer, and conversation transcripts; numeric `context_length`; structured arrays of session metadata; relative paths into the image FileSet. Image *bytes* are NOT part of the instance; only URLs and retrieval metadata are released. Downstream users fetch the images through the bundled download utility.

**Is there a label or target associated with each instance?**
Yes — `reference_answer` is the target. It is consumed by the LLM-as-judge scoring pipeline (and by deterministic-rule normalization for the closed-form subtypes).

**Is any information missing from individual instances?**
No internal-state information is hidden from evaluators; the public release is a complete description of every question. The original raw artefacts (full LLM dialogue traces, raw iCrawler search results, human-annotation worksheets) are retained internally for reproducibility.

**Are relationships between individual instances made explicit?**
Yes — the 789 `question_id`s are stable across the four context-length files, so cross-length analyses are immediate. `evidence_session_ids` and `image_paths` make the per-question evidence localization explicit.

**Are there recommended data splits?**
MEMLENS is *evaluation-only* — it has no train/val/test split. We discourage inclusion of MEMLENS items in any training data. A stratified 195-question subset (`agent_subset_195.json`) is provided as an *indexing file* for memory-agent evaluation; the full 789-question benchmark is the primary evaluation surface and per-type proportions are preserved on the subset to within 0.2 percentage points of the full benchmark.

**Are there any errors, sources of noise, or redundancies in the dataset?**
The 484-question consensus audit identified residual judge-rubric ambiguity in a small fraction of items, documented in the paper appendix. Image retrieval through public web search introduces standard noise (e.g., near-duplicate images, mis-labeled metadata at source); the perceptual-hash and CLIP/SigLIP filters reduce but do not eliminate this.

**Is the dataset self-contained, or does it link to external resources?**
The release links to images via per-image source URLs. Source URLs may decay over time; the recorded retrieval query and CLIP/SigLIP threshold permit semantically equivalent re-retrieval, and a community Hugging Face fallback mirror is maintained for academic redistribution where source license permits.

**Does the dataset contain data that might be considered confidential, offensive, threatening, insulting, harmful, or sensitive?**
MEMLENS dialogues are between two AI agents and contain no real-user content. Images are retrieved from general-purpose web search and may incidentally depict identifiable persons (e.g., public figures in news photos); a watermark/logo/overlay filter is applied at retrieval time, but no face detection is performed. A takedown SLA (seven days) is in place — see Section 6.

**Does the dataset relate to people?**
The dialogues do not. Some images may incidentally include people; see above.

**Does the dataset identify any subpopulations (e.g., by age, gender) and is there any way subpopulations are unfairly represented?**
No subpopulation labels are present. As noted in the Croissant `rai:dataBiases` field, image retrieval inherits the demographic, geographic, and topical biases of general-purpose web image search; the topic ontology reflects the dataset team's selection priors; and synthetic dialogues inherit the Western-centric, English-dominant biases of the underlying frontier LLMs.

**Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly?**
Not from the dialogues. Images may include identifiable public figures incidentally; flagged images are removed from the public manifest within seven days.

**Does the dataset contain data that might be considered sensitive in any way?**
No medical, financial, biometric, or otherwise regulated personal data is included.

---

## 3. Collection Process

**How was the data associated with each instance acquired?**
Three channels: (i) a topic ontology authored by the dataset team; (ii) iCrawler queries against general-purpose web image search, plus the watermark/logo/overlay filter and CLIP/SigLIP relevance scoring; (iii) a two-agent dialogue between GPT-5.1 (user role) and Gemini-3-Pro (assistant role) conditioned on each surviving image-topic pair.

**What mechanisms or procedures were used to collect the data?**
The full pipeline is implemented in the project codebase and reproducible from the released metadata. iCrawler interactions follow the public web image-search APIs; LLM dialogue generation uses the official provider APIs; quality-control filters are documented in the paper appendix.

**If the dataset is a sample, what was the sampling strategy?**
The topic ontology defines the population of question scenarios; image retrieval is a CLIP/SigLIP-thresholded sample of search results; question generation produces multiple candidates per scenario, of which the highest-quality ones (per judge-rubric tractability) are retained; the 484-question human consensus audit further filters the surviving items.

**Who was involved in the data collection process and how were they compensated?**
Human raters who performed the 484-question consensus audit are computer-science-trained, English-fluent researchers recruited from the team's institutional network at HKUST. Each item was rated by at least two raters independently, with a third rater adjudicating disagreements. Raters were compensated at or above the local minimum wage applicable to research assistantships in their jurisdiction.

**Over what timeframe was the data collected?**
The construction pipeline, image retrieval, dialogue generation, and human auditing were completed between September 2025 and May 2026.

**Were any ethical review processes conducted?**
Image retrieval and LLM-generated dialogue do not involve human subjects in the IRB sense. The 484-question rater audit involved only adult researchers in their professional capacity, did not collect personally identifying information from raters, and did not constitute human-subjects research under the relevant institutional definitions.

**Does the dataset relate to people?**
Only incidentally via images. See Sections 2 and 6.

---

## 4. Preprocessing / Cleaning / Labeling

**Was any preprocessing/cleaning/labeling of the data done?**
Yes, three layers:

1. **Per-image filters** — watermark / stock-photo logo / copyright-overlay rejection; CLIP and SigLIP relevance thresholds (specific values reported in the paper appendix).

2. **Per-conversation filters** — structural validation (turn count, image-token alignment between markup and image array), evidence-session uniqueness (no two questions share the same source image), and evidence-position randomization (uniform across haystack except for KU, where evidence order encodes update chronology).

3. **Per-question filters** — judge-rubric tractability checks; deterministic-rule normalization for closed-form subtypes (7 of 9 reporting subtypes), so that judge variance is bounded at scoring time.

Length controls instantiate the same 789 questions at 32K/64K/128K/256K via varying haystack-session counts; cross-modal token counting follows the protocol of MMLongBench.

**Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data?**
Yes, retained internally for reproducibility; not part of the public release. The public release contains the structured derivative (URL manifest + per-question JSON + prompt templates + judge rubrics).

**Is the software used to preprocess/clean/label the instances available?**
Yes — the evaluation harness, judge rubrics, and prompt templates are released alongside the dataset under the MIT license (code) and CC-BY-4.0 (data artefacts).

---

## 5. Uses

**Has the dataset been used for any tasks already?**
Yes — for the analyses reported in the accompanying NeurIPS 2026 paper, which evaluates 28 long-context VLMs and seven memory agents at four context lengths.

**Is there a repository that links to any or all papers or systems that use the dataset?**
The dataset card on Hugging Face is the canonical pointer; once papers begin to cite MemLens, citations will be aggregated there.

**What (other) tasks could the dataset be used for?**
Ablation studies of memory mechanisms; failure-mode analysis under context-length stress; comparison of LC-VLMs versus memory-agent pipelines on a fixed evaluation surface; downstream studies of cross-modal abstention behavior.

**Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?**
Yes — see the Croissant `rai:dataLimitations` and `rai:dataBiases` fields. Headline caveats: synthetic-dialogue distribution does not reflect real-user behavior; English-only; small-cohort human auditing; image retrieval inherits Western/English/Internet biases of general-purpose web search; benchmark gaming risk if MEMLENS items enter training data.

**Are there tasks for which the dataset should not be used?**
Production-system training; deployment-grade safety claims; biometric or surveillance applications; psychometric measurement of real-user behavior; cross-cultural fairness claims without explicit reweighting and reporting of imposed effective coverage.

---

## 6. Distribution

**Will the dataset be distributed to third parties outside of the entity on behalf of which the dataset was created?**
Yes — MEMLENS is a public-release benchmark.

**How will the dataset be distributed?**
Via a Hugging Face repository containing the four `dataset_*.json` files (32K/64K/128K/256K), the `agent_subset_195.json` indexing file, the `metadata/croissant.json` Croissant + RAI manifest, and the `release_images/` directory (4,695 unique images, ~219 MB) referenced by the dataset records. An archival deposit on Zenodo with a permanent DOI will follow at camera-ready.

**When will the dataset be distributed?**
v1.0.0 is tagged at submission (2026-05-06). Subsequent versions, if any, follow semantic-version tags without rewriting historical tags.

**Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use?**
- *Code* (evaluation harness, judge runners, etc.): MIT (see `LICENSE-CODE`).
- *Dataset artefacts* (URL manifest, per-question metadata, prompt templates, judge rubrics, annotation worksheets): CC-BY-4.0 (see `LICENSE-DATA`).
- *Underlying images*: NOT redistributed. Each image's source-side license applies; downstream users honor those terms when they fetch through the bundled download utility.

**Have any third parties imposed IP-based or other restrictions on the data associated with the instances?**
Per-image source-side licenses apply; the public release does not redistribute image bytes, only URLs and retrieval metadata.

**Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?**
None known.

---

## 7. Maintenance

**Who will be supporting/hosting/maintaining the dataset?**
The MemLens team at HKUST, led by Xiyu Ren and Zhaowei Wang. Maintenance is committed for at least one full conference cycle beyond submission (through NeurIPS 2027); extensions are announced on the dataset card.

**How can the owner/curator/manager of the dataset be contacted?**
For general inquiries or image-takedown requests, contact Xiyu Ren at xrenaf@cse.ust.hk. Flagged images will be removed from the public release within seven days.

**Is there an erratum?**
Errata, if any, are tracked on the dataset card; major revisions are released as new tags (v1.1.0, v2.0.0, …) without rewriting historical tags so that prior evaluation runs remain reproducible.

**Will the dataset be updated?**
Updates are infrequent and version-tagged; minor-version updates (v1.x) preserve backward compatibility and the same `question_id` set; major-version updates may revise the question set and are released alongside change-notes.

**If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances?**
Not applicable — dialogues are between two AI agents and contain no real-user data.

**Will older versions of the dataset continue to be supported/hosted/maintained?**
Yes — historical tags are preserved without modification so that prior evaluation runs remain reproducible.

**If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?**
Yes — issue tracker and pull requests on the dataset repository. Extensions are encouraged and may be linked from the dataset card; substantive forks are welcome under the existing licenses.

---

*Datasheet template adapted from Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for Datasets. Communications of the ACM, 64(12), 86-92.*
