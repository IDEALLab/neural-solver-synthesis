# Prerequisites for Running SDS Pipeline

To run `run_sds_pipeline.py`, you need:

## 1. ShinkaEvolve Installed
```bash
# From project root
pip install -e .
# or
uv pip install -e .
```

## 2. API Keys
Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
# Or create a .env file in project root with:
# OPENAI_API_KEY=sk-...
```

## 3. Problem Generation Module
The pipeline requires `gen_sds_dataset.py` module that provides:
- `make_dense_instance(n, seed)` - Generate dense problem instances
- `make_tree_instance(n, seed)` - Generate tree problem instances  
- `_instance_to_problem(inst, ptype, i)` - Convert instance to problem dict
- `sds_render_prompt(problem_dict)` - Render problem as prompt

**This module should be:**
- In your Python path, OR
- In the same directory as `run_sds_pipeline.py`, OR
- Installed as a package

## 4. Optional Dependencies
For pushing to HuggingFace:
```bash
pip install datasets huggingface_hub
```

## Quick Start

Once prerequisites are met:

```bash
cd examples/sds
python run_sds_pipeline.py \
    --samples 10 \
    --generations 5 \
    --output_dir my_dataset
```

The pipeline will:
1. ✅ Generate problem instances automatically
2. ✅ Run ShinkaEvolve evolution for each
3. ✅ Extract best solutions from database
4. ✅ Generate reasoning traces
5. ✅ Create SFT dataset (JSONL + JSON)

## Troubleshooting

**"Could not import gen_sds_dataset"**
- Make sure `gen_sds_dataset.py` is in your Python path
- Or place it in `examples/sds/` directory

**"OpenAI API Key required"**
- Set `OPENAI_API_KEY` environment variable
- Or pass `--api_key sk-...` as argument

