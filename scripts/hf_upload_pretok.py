"""Upload the 3-level pretokenized tars -> HF dataset yuchengm/quadtok_data under pretok_3level/.
Run from the STABLE relay (not the preemptible job). Needs HF_TOKEN (write) in env."""
import os
from huggingface_hub import HfApi

OUT = os.environ.get("OUT", "/sensei-fs-3/users/yuchengm/data/quadtok_pretok_3level")
REPO = "yuchengm/quadtok_data"
api = HfApi(token=os.environ["HF_TOKEN"])

ntar = len([f for f in os.listdir(OUT) if f.endswith(".tar.gz")])
sz = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.endswith(".tar.gz"))
print(f"local: {ntar} .tar.gz, {sz/1e9:.1f} GB in {OUT}")
assert ntar == 147, f"expected 147 .tar.gz, got {ntar}"

api.create_repo(REPO, repo_type="dataset", exist_ok=True)
print(f"uploading {OUT}/*.tar.gz -> {REPO}:pretok_3level/ ...", flush=True)
api.upload_folder(
    folder_path=OUT,
    path_in_repo="pretok_3level",
    repo_id=REPO,
    repo_type="dataset",
    allow_patterns=["*.tar.gz"],
    commit_message="Add 3-level pretokenized ImageNet-train (BFS order, BICUBIC+hflip, int16, gzip)",
)
files = [f for f in api.list_repo_files(REPO, repo_type="dataset") if f.startswith("pretok_3level/")]
print(f"UPLOAD_DONE: {len(files)} files under pretok_3level/ on {REPO}", flush=True)
