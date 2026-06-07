"""
Конвертирует HuggingFace датасет в uint32 numpy memmap для OLMo.
Поддерживает shuffle документов перед сохранением.

Использование:
    uv run python scripts/prepare_hf_dataset.py \
        --dataset allenai/olmo-mix-1124 \
        --output data/olmo-mix/train/ \
        --max-tokens 500_000_000 \
        --shuffle --seed 6198
"""

import argparse
import random
import numpy as np
from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EOS_TOKEN_ID = 100257

# Сколько документов буферизуем перед shuffle.
# Больше → лучше перемешивание, но больше RAM.
# 100_000 документов × ~500 токенов ≈ 200MB RAM.
SHUFFLE_BUFFER = 100_000


def resolve_tokenizer_path(path: str) -> Path:
    """Ищет файл токенизатора: сначала как есть, потом относительно корня проекта."""
    p = Path(path)
    if p.is_file():
        return p
    p_from_root = PROJECT_ROOT / path
    if p_from_root.is_file():
        return p_from_root
    raise FileNotFoundError(
        f"Токенизатор не найден: {path}\n"
        f"Скачайте:\n"
        f"  mkdir -p tokenizers && wget "
        f"https://huggingface.co/allenai/OLMo-2-1124-7B/resolve/main/tokenizer.json "
        f"-O {PROJECT_ROOT / 'tokenizers' / 'allenai_dolma2.json'}"
    )


def flush_shard(tokens: np.ndarray, path: Path) -> None:
    """Записывает numpy uint32 массив в memmap-файл без лишних копий."""
    n = len(tokens)
    fp = np.memmap(path, dtype=np.uint32, mode="w+", shape=(n,))
    fp[:] = tokens
    fp.flush()
    del fp
    print(f"  Шард: {path} ({n:,} токенов, {n * 4 / 1e9:.2f} GB)")


def iter_docs(ds, text_field: str, shuffle: bool, seed: int, shuffle_buffer: int):
    """Итерирует документы, опционально перемешивая буфер."""
    if not shuffle:
        yield from ds
        return

    rng = random.Random(seed)
    buf: list = []
    for doc in ds:
        buf.append(doc)
        if len(buf) >= shuffle_buffer:
            rng.shuffle(buf)
            yield from buf
            buf = []
    if buf:
        rng.shuffle(buf)
        yield from buf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="allenai/olmo-mix-1124")
    parser.add_argument("--subset", default=None, help="Конкретный subset датасета")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--tokenizer",
        default="tokenizers/allenai_dolma2.json",
        help="Путь к tokenizer.json (абсолютный или относительно корня проекта)",
    )
    parser.add_argument("--max-tokens", type=int, default=500_000_000)
    parser.add_argument(
        "--shard-size",
        type=int,
        default=512 * 1024 * 1024,
        help="Токенов на шард (default: 512M = 2 GB)",
    )
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Перемешать документы перед сохранением",
    )
    parser.add_argument("--seed", type=int, default=6198, help="Seed для shuffle")
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=SHUFFLE_BUFFER,
        help="Кол-во документов в буфере shuffle (default: 100_000)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = resolve_tokenizer_path(args.tokenizer)
    print(f"Токенизатор:  {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    print(f"Датасет:      {args.dataset}")
    print(f"Shuffle:      {args.shuffle} (seed={args.seed}, buffer={args.shuffle_buffer:,})")
    print(f"Лимит:        {args.max_tokens:,} токенов")

    ds = load_dataset(
        args.dataset,
        name=args.subset,
        split=args.split,
        streaming=True,
    )

    shard_idx = 0
    total_written = 0
    # Используем numpy-массив вместо list[int]:
    # Python int занимает 28 байт, numpy uint32 — 4 байта.
    # При shard_size=512M разница: 14 GB vs 2 GB.
    token_buf = np.empty(args.shard_size, dtype=np.uint32)
    buf_pos = 0

    for doc in iter_docs(ds, args.text_field, args.shuffle, args.seed, args.shuffle_buffer):
        text = doc.get(args.text_field, "")
        if not text.strip():
            continue

        ids = tokenizer.encode(text).ids + [EOS_TOKEN_ID]
        ids_arr = np.array(ids, dtype=np.uint32)
        total_written += len(ids_arr)

        offset = 0
        while offset < len(ids_arr):
            space = args.shard_size - buf_pos
            chunk = ids_arr[offset : offset + space]
            token_buf[buf_pos : buf_pos + len(chunk)] = chunk
            buf_pos += len(chunk)
            offset += len(chunk)

            if buf_pos >= args.shard_size:
                flush_shard(token_buf[:buf_pos], output_dir / f"{shard_idx:05d}_00000.npy")
                buf_pos = 0
                shard_idx += 1

        if total_written >= args.max_tokens:
            print(f"Достигнут лимит {args.max_tokens:,} токенов.")
            break

    if buf_pos > 0:
        flush_shard(token_buf[:buf_pos], output_dir / f"{shard_idx:05d}_00000.npy")
        shard_idx += 1

    print(f"\nГотово: {shard_idx} шардов, {total_written:,} токенов")
    print("Пути для yaml:")
    for i in range(shard_idx):
        print(f"  - {output_dir / f'{i:05d}_00000.npy'}")


if __name__ == "__main__":
    main()
