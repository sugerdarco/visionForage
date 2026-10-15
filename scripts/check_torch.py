"""Exercise the operations needed by training without downloading a model."""

import argparse
import json
import os
import platform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("Set CUDA_VISIBLE_DEVICES to one allocated GPU index or UUID.")

    import torch
    import torchvision
    from torchvision.ops import nms

    torch.set_num_threads(2)
    torch.manual_seed(42)
    if args.device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one visible, usable CUDA GPU.")
    device = torch.device(args.device)
    # Compare a real device computation with a CPU reference.
    a = torch.randn(32, 32)
    b = torch.randn(32, 32)
    torch.testing.assert_close((a.to(device) @ b.to(device)).cpu(), a @ b)

    model = torch.nn.Conv2d(3, 4, kernel_size=3).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = torch.randn(2, 3, 32, 32, device=device)
    supports_bf16 = device.type == "cpu" or torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if supports_bf16 else torch.float16
    with torch.autocast(device_type=device.type, dtype=dtype):
        loss = model(inputs).float().square().mean()
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError("Backward pass produced missing or non-finite gradients.")
    optimizer.step()

    boxes = torch.tensor(
        [[0, 0, 10, 10], [1, 1, 9, 9]], dtype=torch.float32, device=device
    )
    scores = torch.tensor([0.9, 0.8], device=device)
    if nms(boxes, scores, 0.5).tolist() != [0]:
        raise RuntimeError("torchvision NMS returned an unexpected result.")
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(json.dumps(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "mixed_precision": str(dtype),
            "checks": "matmul, convolution, backward, optimizer, NMS passed",
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
