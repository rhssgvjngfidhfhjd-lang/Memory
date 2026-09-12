# VP Extractor

Open-vocabulary visual primitive discovery and cropping with
`Qwen/Qwen3-VL-4B-Instruct`.

The core API processes one canonicalized RGB image at a time. Dataset scanning,
artifact storage, and JSONL export are thin outer layers, so the extractor can
also be reused by other memory pipelines.

## Install

```bash
cd vp_extractor
python -m pip install -e .
```

## Check the local model

```bash
vp-extractor check-model
```

The default endpoint is `http://127.0.0.1:18000/v1` and can be overridden:

```bash
vp-extractor --base-url http://127.0.0.1:18001/v1 check-model
```

## Extract

One image or directory:

```bash
vp-extractor extract --input path/to/image.jpg
vp-extractor extract --input path/to/images --dataset-name custom
```

Mem-Gallery images can use their dialog captions to keep only the main visual
depiction and ignore repeated thumbnails or alternate views:

```bash
vp-extractor extract --input path/to/category/images \
  --dataset-name Mem-Gallery --caption-file path/to/category.json
```

Inputs without a caption keep using generic image-only discovery. The
configured `Mem-Gallery` dataset loads captions automatically.

A configured benchmark:

```bash
vp-extractor extract --dataset H2HMEM --limit 10
vp-extractor extract --dataset Mem-Gallery
vp-extractor extract --dataset WorldMemArena
vp-extractor extract --dataset all
```

Large runs can be split deterministically across workers. All shards may share
the same run directory because each source image owns a distinct item folder:

```bash
vp-extractor extract --dataset all --num-shards 8 --shard-index 0
```

Results are written to `outputs/<run_id>/`. Existing `record.json` files are
skipped unless `--force` is supplied. JSONL exports are rebuilt at the end of
each run and can also be regenerated independently:

```bash
vp-extractor export
```

## Output contract

Each source image has one `items/<image_id>/record.json` and zero or more VP
crops. A primitive contains a VLM-generated open-vocabulary `label`, a
`bbox_norm` in Qwen's 0..1000 coordinate space, the corresponding half-open
pixel `bbox_px`, and its crop path. `record.json` is canonical; files under
`exports/` are derived indexes.

## Tests

Tests use fake VLM responses and do not require a running model server:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
