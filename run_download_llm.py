### This script download and locally save Large Language Model (LLM) models from HuggingFace
from huggingface_hub import snapshot_download
import os


def llm_downloader(args):
    """
    Download a LLM model from HuggingFace and save it locally.
    """
    local_dir = os.path.join(args.local_dir, args.repo_id.split("/")[-1].replace(".", "_"))
    os.makedirs(local_dir, exist_ok=False)
    returned_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        local_dir=local_dir,
        local_dir_use_symlinks=args.local_dir_use_symlinks,
        revision=args.revision,
        token=args.authentication_token,
        allow_patterns=args.allow_patterns
    )
    print(f"Downloaded {args.repo_id} to {returned_path}")


if __name__ == "__main__":
    import argparse
    from config import LANG_DIR
    parser = argparse.ArgumentParser(description="Download Large Language Models (LLM) from HuggingFace")
    parser.add_argument("--repo_id", type=str,required=True, help="The HuggingFace repo id")
    parser.add_argument("--repo_type", type=str, default=None, required=False, help="The HuggingFace repo type")
    parser.add_argument("--revision", type=str, default=None, required=False, help="revision of the repo")
    parser.add_argument("--local_dir", type=str, required=False, default=LANG_DIR, help="The local directory to save the file")
    parser.add_argument("--local_dir_use_symlinks", type=bool, default=None, required=False, help="Use symlinks to save the file")
    parser.add_argument("--authentication_token", type=str, default=None, required=False, help="Hugging face authentication token. None, try to read from env")
    parser.add_argument("--allow_patterns", type=str, default=None, required=False, help="Huggingface cli reqex patterns ")
    args = parser.parse_args()
    llm_downloader(args)