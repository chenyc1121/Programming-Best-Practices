import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm
from transformers import AutoImageProcessor, ResNetConfig, ResNetForImageClassification


MODEL_ID = "microsoft/resnet-18"
SEED = 42
BATCH_SIZE = 64
PROJECT_ROOT = Path(__file__).resolve().parent


def collate_batch(batch, processor):
    """Convert grayscale MNIST to RGB and resize/normalize for ResNet."""
    images = [image.convert("RGB") for image, _ in batch]
    labels = torch.tensor([label for _, label in batch], dtype=torch.long)
    inputs = processor(images=images, do_resize=True, return_tensors="pt")
    return inputs["pixel_values"], labels


def load_model(cache_dir):
    """Load pretrained backbone weights and initialize a new 10-digit head."""
    id2label = {digit: str(digit) for digit in range(10)}
    config = ResNetConfig.from_pretrained(MODEL_ID, cache_dir=str(cache_dir))
    config.id2label = id2label
    config.label2id = {label: digit for digit, label in id2label.items()}
    config.num_labels = 10
    return ResNetForImageClassification.from_pretrained(
        MODEL_ID,
        cache_dir=str(cache_dir),
        config=config,
        ignore_mismatched_sizes=True,
    )


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = PROJECT_ROOT / ".cache" / "huggingface"
    print(f"Device: {device}", flush=True)
    print(f"Loading pretrained ResNet: {MODEL_ID}", flush=True)

    processor = AutoImageProcessor.from_pretrained(
        MODEL_ID, cache_dir=str(cache_dir)
    )
    model = load_model(cache_dir).to(device)
    model.eval()

    dataset = MNIST(root=str(PROJECT_ROOT / "data"), train=False, download=True)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=partial(collate_batch, processor=processor),
    )

    correct = 0
    total = 0
    with torch.inference_mode():
        for pixel_values, labels in tqdm(loader, desc="MNIST inference"):
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(pixel_values=pixel_values).logits
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()

    accuracy_percent = 100.0 * correct / total
    metrics = {
        "model_id": MODEL_ID,
        "experiment": "ImageNet-pretrained backbone with random 10-class head",
        "fine_tuned": False,
        "seed": SEED,
        "device": str(device),
        "samples": total,
        "correct": correct,
        "accuracy_percent": accuracy_percent,
    }
    metrics_path = PROJECT_ROOT / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"MNIST accuracy: {accuracy_percent:.2f}% ({correct}/{total})")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
