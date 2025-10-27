
export PYTHONPATH=$PYTHONPATH:$PWD
espeak --version

/mnt/juicefs/user/jiangyuepeng1/miniforge3/envs/os/bin/python inference.py \
    --repo-id ASLP-lab/DiffRhythm2 \
    --output-dir ./results/test \
    --input-jsonl ./example/test.jsonl \
    --cfg-strength 2.0 \
