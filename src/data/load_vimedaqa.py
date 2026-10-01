"""
Load the raw ViMedAQA dataset from HuggingFace or local disk.

Responsibilities
----------------
- Download and cache the dataset via the HuggingFace `datasets` library.
- Save the raw dataset to `data/raw/vimedaqa` for offline use.
- Return the raw DatasetDict (train / validation / test splits).
- No filtering, deduplication, or file I/O for processing — that belongs in data_processing.py.

Usage:
    from src.data.load_vimedaqa import load_vimedaqa

    ds = load_vimedaqa()
    print(ds)                  # DatasetDict
    print(ds["train"][0])      # first raw row
"""

from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk
from loguru import logger

DATASET_ID = "tmnam20/ViMedAQA"
CONFIG     = "all"   # loads all 4 topics in one DatasetDict
RAW_DATA_DIR = Path("data/raw/vimedaqa")


def load_vimedaqa(dataset_id: str = DATASET_ID, save_dir: Path | str = RAW_DATA_DIR) -> DatasetDict:
    """
    Download and return the raw ViMedAQA DatasetDict, saving it locally.
    If the dataset already exists locally, it loads from disk instead.

    The dataset has three splits: train, validation, test.
    Each row contains:
        question_idx, question, answer, context,
        title, keyword, topic, article_url, author, author_url

    Topic ClassLabel:
        0 = body-part  |  1 = disease  |  2 = drug  |  3 = medicine

    Parameters
    ----------
    dataset_id : HuggingFace dataset repository ID
    save_dir   : Local directory to save/load the raw dataset

    Returns
    -------
    DatasetDict with keys 'train', 'validation', 'test'
    """
    save_path = Path(save_dir)
    
    if save_path.exists():
        logger.info(f"Loading '{dataset_id}' from local disk: {save_path} …")
        ds = load_from_disk(str(save_path))
    else:
        logger.info(f"Loading '{dataset_id}' from HuggingFace Hub …")
        ds = load_dataset(dataset_id, CONFIG, trust_remote_code=True)
        
        logger.info(f"Saving raw dataset to {save_path} …")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(save_path))
        
    logger.success(
        f"Loaded — "
        f"train: {len(ds['train']):,} | "
        f"val: {len(ds['validation']):,} | "
        f"test: {len(ds['test']):,}"
    )
    return ds


if __name__ == "__main__":
    ds = load_vimedaqa()

    # Quick inspection
    logger.info("Sample row (train[0])")
    row = ds["train"][0]
    for key, val in row.items():
        preview = str(val)[:120].replace("\n", " ")
        logger.info(f"  {key:<15}: {preview}")
