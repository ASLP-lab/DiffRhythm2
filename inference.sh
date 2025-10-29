
export PYTHONPATH=$PYTHONPATH:$PWD
espeak-ng --version

python inference.py \
    --repo-id ASLP-lab/DiffRhythm2 \
    --output-dir ./results/test \
    --input-jsonl ./example/test.jsonl \
    --cfg-strength 2.0 \
