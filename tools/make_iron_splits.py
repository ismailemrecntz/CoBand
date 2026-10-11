"""Create train.txt and test.txt for the IRON dataset (sequence split of the original IRON paper).

    Train (27 sequences, 20,022 frames): countryside 1-10, forest 13-25, high 30-33
    Test  (8 sequences,   4,292 frames): countryside 11-12, forest 26-29, high 34-35

Each line is scene/sequence/frame (without extension), e.g. countryside/1/image_000000_1746839541.889976711

    python tools/make_iron_splits.py --data_root /path/to/IRON_Dataset
"""
import argparse
import os

TRAIN_SEQUENCES = {
    "countryside": list(range(1, 11)),
    "forest": list(range(13, 26)),
    "high": list(range(30, 34)),
}
TEST_SEQUENCES = {
    "countryside": list(range(11, 13)),
    "forest": list(range(26, 30)),
    "high": list(range(34, 36)),
}


def collect_samples(data_root, split_sequences):
    samples = []
    for scene, seq_ids in split_sequences.items():
        for seq_id in seq_ids:
            img_dir = os.path.join(data_root, "images", scene, str(seq_id))
            if not os.path.isdir(img_dir):
                print(f"  [WARNING] Directory not found: {img_dir}")
                continue
            for f in sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg")):
                samples.append(f"{scene}/{seq_id}/{os.path.splitext(f)[0]}")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Create IRON split files")
    parser.add_argument("--data_root", type=str, required=True, help="root directory of the IRON dataset")
    args = parser.parse_args()

    train_samples = collect_samples(args.data_root, TRAIN_SEQUENCES)
    test_samples = collect_samples(args.data_root, TEST_SEQUENCES)
    print(f"train: {len(train_samples)} frames, test: {len(test_samples)} frames")
    assert len(train_samples) == 20022, f"Expected 20022 train frames, got {len(train_samples)}"
    assert len(test_samples) == 4292, f"Expected 4292 test frames, got {len(test_samples)}"

    for name, samples in (("train", train_samples), ("test", test_samples)):
        path = os.path.join(args.data_root, f"{name}.txt")
        with open(path, "w") as f:
            f.writelines(s + "\n" for s in samples)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
