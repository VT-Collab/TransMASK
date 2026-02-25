from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="SagarParekh/TransMASK",
    repo_type="dataset",
    allow_patterns="datasets/*",
    local_dir="./",
)