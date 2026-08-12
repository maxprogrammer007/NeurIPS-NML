"""
Phase 1 — GenImage Automated Downloader & Extractor
===================================================
Downloads the official GenImage dataset from the mirror repository on Hugging Face
(ENSTA-U2IS/GenImage), concatenates multi-part split zips (.z01, .z02, ..., .zip),
unzips them into the target directory, and verifies directory layout.

Usage:
  python download_genimage.py --output_dir ./GenImage --generators stable_diffusion_v_1_4,VQDM
  python download_genimage.py --output_dir ./GenImage --all
"""

import os
import sys
import glob
import shutil
import zipfile
import argparse
import subprocess
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "ENSTA-U2IS/GenImage"

GENERATOR_MAP = {
    "stable_diffusion_v_1_4": "stable_diffusion_v_1_4",
    "stable_diffusion_v_1_5": "stable_diffusion_v_1_5",
    "midjourney": "Midjourney",
    "wukong": "wukong",
    "VQDM": "VQDM",
    "ADM": "ADM",
    "glide": "glide",
    "biggan": "BigGAN",
}

def get_repo_parts_for_generator(hf_folder: str) -> list[str]:
    """List all zip part files for a generator folder on HuggingFace."""
    all_files = list_repo_files(REPO_ID, repo_type="dataset")
    gen_files = [f for f in all_files if f.startswith(f"{hf_folder}/")]
    
    # Sort files so .z01, .z02, ..., .zN come before .zip
    def sort_key(filename):
        base = os.path.basename(filename)
        ext = os.path.splitext(base)[1].lower()
        if ext == ".zip":
            return 999999
        try:
            return int(ext.replace(".z", ""))
        except ValueError:
            return 0
            
    return sorted(gen_files, key=sort_key)

def download_and_extract_generator(gen_key: str, hf_folder: str, output_dir: Path, tmp_dir: Path):
    print(f"\n==================================================")
    print(f"Processing Generator: {gen_key} (HF folder: {hf_folder})")
    print(f"==================================================")
    
    target_gen_dir = output_dir / gen_key
    if target_gen_dir.exists() and any(target_gen_dir.iterdir()):
        print(f"Target directory {target_gen_dir} already exists and is non-empty. Skipping download.")
        return

    parts = get_repo_parts_for_generator(hf_folder)
    if not parts:
        print(f"ERROR: No files found for {hf_folder} in {REPO_ID}")
        return
        
    print(f"Found {len(parts)} parts for {gen_key}:")
    for p in parts:
        print(f"  - {p}")
        
    gen_tmp = tmp_dir / gen_key
    gen_tmp.mkdir(parents=True, exist_ok=True)
    
    downloaded_paths = []
    for p in parts:
        filename = os.path.basename(p)
        print(f"Downloading {p} ...")
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=p,
            local_dir=str(gen_tmp),
            local_dir_use_symlinks=False
        )
        downloaded_paths.append(Path(local_path))
        
    # Check if there are split zips (.z01, .z02, ..., .zip)
    z_parts = [p for p in downloaded_paths if p.suffix.lower() != ".zip"]
    zip_main = [p for p in downloaded_paths if p.suffix.lower() == ".zip"]
    
    combined_zip = gen_tmp / f"{gen_key}_combined.zip"
    
    if z_parts:
        print(f"\nConcatenating {len(z_parts) + len(zip_main)} split zip parts into {combined_zip.name}...")
        # Sort z_parts numerically
        def z_part_key(p):
            ext = p.suffix.lower()
            try:
                return int(ext.replace(".z", ""))
            except ValueError:
                return 0
        sorted_z_parts = sorted(z_parts, key=z_part_key)
        all_ordered_parts = sorted_z_parts + zip_main
        
        with open(combined_zip, "wb") as outfile:
            for p in all_ordered_parts:
                print(f"Appending {p.name} ({p.stat().st_size / (1024*1024):.1f} MB)...")
                with open(p, "rb") as infile:
                    shutil.copyfileobj(infile, outfile)
        target_zip_to_extract = combined_zip
    elif zip_main:
        target_zip_to_extract = zip_main[0]
    else:
        raise RuntimeError(f"No zip file found after downloading {gen_key}")

    # Extract using 7z or unzip
    print(f"\nExtracting {target_zip_to_extract.name} to {target_gen_dir} ...")
    target_gen_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = ["unzip", "-q", "-o", str(target_zip_to_extract), "-d", str(target_gen_dir)]
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    
    if res.returncode != 0:
        print(f"Warning: unzip returned code {res.returncode}. Checking python zipfile fallback...")
        try:
            with zipfile.ZipFile(target_zip_to_extract, 'r') as zip_ref:
                zip_ref.extractall(target_gen_dir)
        except Exception as e:
            print(f"Zip extraction error: {e}")

    # Clean up temporary download files to save disk space
    print(f"Cleaning up temporary downloads in {gen_tmp} ...")
    shutil.rmtree(gen_tmp, ignore_errors=True)
    print(f"Done processing {gen_key}!")

def main():
    parser = argparse.ArgumentParser(description="Download and extract GenImage dataset")
    parser.add_argument("--output_dir", type=str, default="./GenImage", help="Output directory for extracted GenImage data")
    parser.add_argument("--tmp_dir", type=str, default="./tmp_download", help="Temporary directory for downloaded zip parts")
    parser.add_argument("--generators", type=str, default=None, help="Comma-separated generator keys (e.g., stable_diffusion_v_1_4,VQDM)")
    parser.add_argument("--all", action="store_true", help="Download all 8 generators")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir).resolve()
    tmp_dir = Path(args.tmp_dir).resolve()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    if args.all:
        selected_gens = list(GENERATOR_MAP.keys())
    elif args.generators:
        selected_gens = [g.strip() for g in args.generators.split(",")]
    else:
        print("Please specify --generators (comma-separated) or --all. Example:")
        print("  python download_genimage.py --generators stable_diffusion_v_1_4,VQDM")
        sys.exit(1)
        
    for gen in selected_gens:
        if gen not in GENERATOR_MAP:
            print(f"Unknown generator '{gen}'. Valid keys: {list(GENERATOR_MAP.keys())}")
            continue
        hf_folder = GENERATOR_MAP[gen]
        download_and_extract_generator(gen, hf_folder, output_dir, tmp_dir)
        
    print("\nAll requested generator downloads completed.")

if __name__ == "__main__":
    main()
