from __future__ import annotations

import argparse
import copy
import inspect
import json
import logging
import math
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import v2 as transforms_v2
from PIL import Image
from tqdm.auto import tqdm

from .data import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    VSDLMDataset,
    build_weighted_sampler,
    collect_samples,
    create_dataloader,
    split_samples,
)
from .model import VSDLM, ModelConfig

LOGGER = logging.getLogger("vsdlm")

LABEL_MAP = {0: "closed", 1: "open"}


if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):

    _grad_scaler_params = inspect.signature(torch.amp.GradScaler).parameters

    if "device" in _grad_scaler_params:

        def _create_grad_scaler(enabled: bool):
            return torch.amp.GradScaler("cuda", enabled=enabled)

    elif "device_type" in _grad_scaler_params:

        def _create_grad_scaler(enabled: bool):
            return torch.amp.GradScaler(device_type="cuda", enabled=enabled)

    else:

        def _create_grad_scaler(enabled: bool):
            return torch.amp.GradScaler(enabled=enabled)

    _autocast_params = inspect.signature(torch.amp.autocast).parameters

    if "device_type" in _autocast_params:

        def _autocast(enabled: bool):
            if not enabled:
                return nullcontext()
            return torch.amp.autocast(device_type="cuda", enabled=True)

    else:

        def _autocast(enabled: bool):
            if not enabled:
                return nullcontext()
            return torch.amp.autocast("cuda", enabled=True)


else:
    from torch.cuda.amp import GradScaler as _CudaGradScaler
    from torch.cuda.amp import autocast as _cuda_autocast

    def _create_grad_scaler(enabled: bool):
        return _CudaGradScaler(enabled=enabled)

    def _autocast(enabled: bool):
        if not enabled:
            return nullcontext()
        return _cuda_autocast(enabled=True)


class RandomCLAHE:
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8), p: float = 0.01) -> None:
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tile_grid_size
        self.p = float(p)

    def __call__(self, img: Image.Image) -> Image.Image:
        if torch.rand(1).item() >= self.p:
            return img
        np_img = np.array(img)
        lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge((l_channel, a_channel, b_channel))
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb)


class _BatchNormAffine(nn.Module):
    def __init__(self, scale: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = [1] * x.dim()
        if len(shape) >= 2:
            shape[1] = -1
        return x * self.scale.view(*shape) + self.bias.view(*shape)


def _decompose_batchnorms(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            running_mean = child.running_mean.detach()
            running_var = child.running_var.detach()
            if child.affine:
                weight = child.weight.detach()
                bias = child.bias.detach()
            else:
                weight = torch.ones_like(running_mean)
                bias = torch.zeros_like(running_mean)
            scale = weight / torch.sqrt(running_var + child.eps)
            bias_term = bias - running_mean * scale
            affine = _BatchNormAffine(scale, bias_term)
            setattr(module, name, affine)
        else:
            _decompose_batchnorms(child)


def _remove_batchnorm_from_onnx(model):
    from onnx import helper, numpy_helper

    graph = model.graph
    initializer_map = {init.name: init for init in graph.initializer}
    value_map = {name: numpy_helper.to_array(init) for name, init in initializer_map.items()}

    additional_initializers = []
    removed_initializers = set()
    new_nodes = []

    for node in graph.node:
        if node.op_type != "BatchNormalization":
            new_nodes.append(node)
            continue

        if len(node.input) < 5:
            new_nodes.append(node)
            continue

        inputs = node.input
        if any(name not in value_map for name in inputs[1:5]):
            new_nodes.append(node)
            continue

        eps = 1e-5
        for attr in node.attribute:
            if attr.name == "epsilon":
                eps = attr.f
                break

        scale = value_map[inputs[1]].astype(np.float32)
        bias = value_map[inputs[2]].astype(np.float32)
        mean = value_map[inputs[3]].astype(np.float32)
        var = value_map[inputs[4]].astype(np.float32)

        denom = np.sqrt(var + eps).astype(np.float32)
        alpha = (scale / denom).astype(np.float32)
        beta = (bias - mean * alpha).astype(np.float32)

        alpha_name = f"{node.output[0]}_bn_alpha"
        beta_name = f"{node.output[0]}_bn_beta"
        alpha_init = numpy_helper.from_array(alpha, name=alpha_name)
        beta_init = numpy_helper.from_array(beta, name=beta_name)
        additional_initializers.append(alpha_init)
        additional_initializers.append(beta_init)

        mul_out = f"{node.output[0]}_mul"
        mul_node = helper.make_node(
            "Mul",
            [inputs[0], alpha_name],
            [mul_out],
            name=f"{node.name}_Mul" if node.name else "",
        )
        add_node = helper.make_node(
            "Add",
            [mul_out, beta_name],
            [node.output[0]],
            name=f"{node.name}_Add" if node.name else "",
        )
        new_nodes.extend([mul_node, add_node])
        removed_initializers.update(inputs[1:5])

    graph.ClearField("node")
    graph.node.extend(new_nodes)

    remaining_initializers = [init for init in graph.initializer if init.name not in removed_initializers]
    existing_names = {init.name for init in remaining_initializers}
    for init in additional_initializers:
        if init.name not in existing_names:
            remaining_initializers.append(init)
            existing_names.add(init.name)
    graph.ClearField("initializer")
    graph.initializer.extend(remaining_initializers)

    return model


@dataclass
class TrainConfig:
    data_root: Path
    output_dir: Path
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    image_size: int = 112
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    test_ratio: float = 0.0
    seed: int = 42
    base_channels: int = 32
    num_blocks: int = 4
    dropout: float = 0.3
    device: str = "auto"
    resume_from: Optional[Path] = None
    use_amp: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["data_root"] = str(self.data_root)
        data["output_dir"] = str(self.output_dir)
        if self.resume_from is not None:
            data["resume_from"] = str(self.resume_from)
        return data


def _resolve_device(device_spec: str) -> torch.device:
    if device_spec and device_spec.lower() not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Unsupported device specifier: {device_spec}")
    if device_spec is None or device_spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_spec == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_spec)


def _setup_logging(output_dir: Path, verbose: bool) -> None:
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")

    if LOGGER.handlers:
        for handler in list(LOGGER.handlers):
            LOGGER.removeHandler(handler)
            handler.close()

    handlers: List[logging.Handler] = []

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    handlers.append(console)

    output_dir.mkdir(parents=True, exist_ok=True)
    logfile = output_dir / "train.log"
    file_handler = logging.FileHandler(logfile, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    handlers.append(file_handler)

    for handler in handlers:
        LOGGER.addHandler(handler)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_transforms(image_size: int, mean: Sequence[float], std: Sequence[float]):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms_v2.RandomPhotometricDistort(p=0.5),
            RandomCLAHE(p=0.01, tile_grid_size=(4, 4)),
            transforms.RandomGrayscale(p=0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def _compute_pos_weight(samples: Sequence) -> torch.Tensor:
    positives = sum(sample.label for sample in samples)
    negatives = len(samples) - positives
    if positives == 0 or negatives == 0:
        LOGGER.warning("Cannot compute class-balanced pos_weight (pos=%d, neg=%d). Using 1.0.", positives, negatives)
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(negatives / positives, dtype=torch.float32)


def _infer_accuracy(train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]]) -> float:
    if val_metrics is not None:
        acc = val_metrics.get("accuracy")
        if acc is not None and not math.isnan(acc):
            return float(acc)
    return float(train_metrics.get("accuracy", 0.0))


def _prune_checkpoints(directory: Path, prefix: str, max_keep: int) -> None:
    checkpoints = sorted(directory.glob(f"{prefix}*.pt"))
    if len(checkpoints) <= max_keep:
        return
    for path in checkpoints[:-max_keep]:
        try:
            path.unlink()
        except FileNotFoundError:
            continue


def _run_epoch(
    model: nn.Module,
    dataloader: Optional[DataLoader],
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[Any] = None,
    autocast_enabled: bool = False,
    progress_desc: Optional[str] = None,
) -> Dict[str, float]:
    if dataloader is None or len(dataloader.dataset) == 0:
        return {"loss": float("nan"), "accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}

    train_mode = optimizer is not None
    model.train(mode=train_mode)

    stats = {"loss": 0.0, "samples": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}

    iterator = dataloader
    if progress_desc:
        iterator = tqdm(iterator, desc=progress_desc, leave=False, dynamic_ncols=True)

    for batch in iterator:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with _autocast(autocast_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        if train_mode:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        stats["loss"] += loss.detach().item() * batch_size
        stats["samples"] += batch_size

        preds = (torch.sigmoid(logits) >= 0.5).long()
        labels_int = labels.long()
        stats["tp"] += ((preds == 1) & (labels_int == 1)).sum().item()
        stats["tn"] += ((preds == 0) & (labels_int == 0)).sum().item()
        stats["fp"] += ((preds == 1) & (labels_int == 0)).sum().item()
        stats["fn"] += ((preds == 0) & (labels_int == 1)).sum().item()

    assert stats["samples"] > 0, "No samples processed during epoch."

    avg_loss = stats["loss"] / stats["samples"]
    accuracy = (stats["tp"] + stats["tn"]) / stats["samples"]
    precision = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0.0
    recall = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _evaluate_predictions(model: nn.Module, dataloader: DataLoader, device: torch.device) -> List[Dict[str, Any]]:
    model.eval()
    results: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits)
            for idx in range(images.size(0)):
                results.append(
                    {
                        "path": batch["path"][idx],
                        "video_name": batch["video_name"][idx],
                        "base_frame": batch["base_frame"][idx],
                        "label": int(batch["label"][idx].item()),
                        "logit": float(logits[idx].detach().cpu().item()),
                        "prob_open": float(probs[idx].detach().cpu().item()),
                    }
                )
    return results


def train_pipeline(config: TrainConfig, verbose: bool = False) -> Dict[str, Any]:
    _setup_logging(config.output_dir, verbose=verbose)
    config_dict = config.to_dict()
    train_config_serialized = copy.deepcopy(config_dict)
    LOGGER.info("Starting training with config: %s", json.dumps(config_dict, indent=2))

    device = _resolve_device(config.device)
    LOGGER.info("Using device: %s", device)
    _set_seed(config.seed)
    amp_enabled = bool(config.use_amp and device.type == "cuda")
    if config.use_amp and not amp_enabled:
        LOGGER.warning("--use_amp requested but CUDA device is not available; proceeding without AMP.")
    tb_dir = config.output_dir
    tb_writer = SummaryWriter(log_dir=str(tb_dir))
    history_path = config.output_dir / "history.json"
    history: List[Dict[str, Any]] = []
    if history_path.exists():
        try:
            with open(history_path, "r", encoding="utf-8") as fp:
                loaded_history = json.load(fp)
                if isinstance(loaded_history, list):
                    history = loaded_history
        except Exception as exc:
            LOGGER.warning("Failed to load existing history from %s: %s", history_path, exc)
            history = []

    samples = collect_samples(config.data_root, logger=LOGGER)
    splits = split_samples(
        samples,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
        logger=LOGGER,
    )

    mean, std = DEFAULT_MEAN, DEFAULT_STD
    train_transform, eval_transform = _build_transforms(config.image_size, mean, std)
    normalization = {"mean": list(mean), "std": list(std), "image_size": config.image_size}

    train_dataset = VSDLMDataset(splits["train"], transform=train_transform)
    val_dataset = VSDLMDataset(splits["val"], transform=eval_transform) if splits["val"] else None
    test_dataset = VSDLMDataset(splits["test"], transform=eval_transform) if splits["test"] else None

    train_sampler = build_weighted_sampler(splits["train"])
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
    )
    val_loader = (
        create_dataloader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
        if val_dataset is not None and len(val_dataset) > 0
        else None
    )
    test_loader = (
        create_dataloader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
        if test_dataset is not None and len(test_dataset) > 0
        else None
    )

    model_config = ModelConfig(
        base_channels=config.base_channels,
        num_blocks=config.num_blocks,
        dropout=config.dropout,
    )
    model = VSDLM(model_config).to(device)
    base_metadata = {
        "model_config": asdict(model_config),
        "train_config": train_config_serialized,
        "normalization": normalization,
        "label_map": LABEL_MAP,
        "amp_enabled": amp_enabled,
        "tensorboard_logdir": str(tb_dir),
    }

    pos_weight = _compute_pos_weight(splits["train"]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = _create_grad_scaler(amp_enabled)

    start_epoch = 1
    best_state: Optional[Dict[str, Any]] = None
    best_val_loss = math.inf
    best_checkpoint_path: Optional[Path] = None

    if config.resume_from:
        resume_path = config.resume_from
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_payload = torch.load(resume_path, map_location=device)
        resume_epoch = int(resume_payload.get("epoch", 0))
        LOGGER.info("Resuming from checkpoint %s (epoch %d).", resume_path, resume_epoch)

        checkpoint_norm = resume_payload.get("normalization")
        if checkpoint_norm and checkpoint_norm != normalization:
            LOGGER.warning(
                "Checkpoint normalization %s differs from current settings %s.",
                checkpoint_norm,
                normalization,
            )

        model.load_state_dict(resume_payload["model_state"])
        if resume_payload.get("optimizer_state"):
            optimizer.load_state_dict(resume_payload["optimizer_state"])
        if resume_payload.get("scheduler_state"):
            scheduler.load_state_dict(resume_payload["scheduler_state"])
        if resume_payload.get("scaler_state") and amp_enabled:
            scaler.load_state_dict(resume_payload["scaler_state"])

        start_epoch = resume_epoch + 1
        resume_train_metrics = copy.deepcopy(resume_payload.get("train_metrics") or {})
        resume_val_metrics = copy.deepcopy(resume_payload.get("val_metrics") or None)
        best_val_loss = resume_payload.get("best_val_loss")
        if best_val_loss is None:
            best_val_loss = (
                resume_val_metrics["loss"]
                if isinstance(resume_val_metrics, dict) and "loss" in resume_val_metrics
                else math.inf
            )
        best_accuracy = resume_payload.get("best_accuracy")
        if best_accuracy is None:
            best_accuracy = _infer_accuracy(
                resume_train_metrics,
                resume_val_metrics if isinstance(resume_val_metrics, dict) else None,
            )

        best_state = {
            "epoch": resume_payload.get("best_epoch", resume_epoch),
            "model_state": copy.deepcopy(resume_payload["model_state"]),
            "optimizer_state": copy.deepcopy(resume_payload.get("optimizer_state")),
            "scheduler_state": copy.deepcopy(resume_payload.get("scheduler_state")),
            "scaler_state": copy.deepcopy(resume_payload.get("scaler_state")),
            "train_metrics": resume_train_metrics,
            "val_metrics": resume_val_metrics,
            "best_val_loss": best_val_loss,
            "best_accuracy": best_accuracy,
        }
        best_checkpoint_path = str(resume_path)

        if history:
            history = [entry for entry in history if entry.get("epoch", 0) < start_epoch]

        if start_epoch > config.epochs:
            LOGGER.info(
                "Checkpoint epoch %d exceeds requested total epochs %d; no additional training will be performed.",
                start_epoch - 1,
                config.epochs,
            )

    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler=scaler,
            autocast_enabled=amp_enabled,
            progress_desc=f"Train {epoch}/{config.epochs}",
        )
        val_metrics = (
            _run_epoch(
                model,
                val_loader,
                criterion,
                device,
                optimizer=None,
                scaler=None,
                autocast_enabled=amp_enabled,
                progress_desc=f"Val   {epoch}/{config.epochs}",
            )
            if val_loader
            else None
        )

        if val_metrics:
            scheduler.step(val_metrics["loss"])
        else:
            scheduler.step(train_metrics["loss"])

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics or {"loss": float("nan"), "accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")},
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        LOGGER.info(
            "Epoch %d | train loss %.4f f1 %.3f | val loss %s f1 %s",
            epoch,
            train_metrics["loss"],
            train_metrics["f1"],
            f"{record['val']['loss']:.4f}" if val_metrics else "n/a",
            f"{record['val']['f1']:.3f}" if val_metrics else "n/a",
        )

        tb_writer.add_scalar("loss/train", train_metrics["loss"], epoch)
        tb_writer.add_scalar("metrics/train_accuracy", train_metrics["accuracy"], epoch)
        tb_writer.add_scalar("metrics/train_f1", train_metrics["f1"], epoch)
        if val_metrics is not None:
            tb_writer.add_scalar("loss/val", val_metrics["loss"], epoch)
            tb_writer.add_scalar("metrics/val_accuracy", val_metrics["accuracy"], epoch)
            tb_writer.add_scalar("metrics/val_f1", val_metrics["f1"], epoch)
        tb_writer.flush()

        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()

        epoch_payload = {
            **base_metadata,
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "scaler_state": scaler.state_dict() if amp_enabled else None,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        epoch_path = config.output_dir / f"vsdlm_epoch_{epoch:04d}.pt"
        torch.save(epoch_payload, epoch_path)
        _prune_checkpoints(config.output_dir, "vsdlm_epoch_", 10)

        current_val_loss = val_metrics["loss"] if val_metrics else train_metrics["loss"]
        if current_val_loss < best_val_loss:
            accuracy_value = _infer_accuracy(train_metrics, val_metrics)
            best_val_loss = current_val_loss
            best_state = {
                "epoch": epoch,
                "model_state": copy.deepcopy(model_state),
                "optimizer_state": copy.deepcopy(optimizer_state),
                "scheduler_state": copy.deepcopy(scheduler_state),
                "scaler_state": copy.deepcopy(scaler.state_dict()) if amp_enabled else None,
                "train_metrics": copy.deepcopy(train_metrics),
                "val_metrics": copy.deepcopy(val_metrics) if val_metrics is not None else None,
                "best_val_loss": current_val_loss,
                "best_accuracy": accuracy_value,
            }
            best_checkpoint = dict(epoch_payload)
            best_checkpoint.update(
                best_val_loss=current_val_loss,
                best_epoch=epoch,
                best_accuracy=accuracy_value,
            )
            best_checkpoint_path = config.output_dir / f"vsdlm_best_epoch{epoch:04d}_acc{accuracy_value:.4f}.pt"
            best_state["checkpoint_path"] = str(best_checkpoint_path)
            torch.save(best_checkpoint, best_checkpoint_path)
            _prune_checkpoints(config.output_dir, "vsdlm_best_", 10)
            LOGGER.info("New best model at epoch %d (loss %.4f).", epoch, current_val_loss)

    if best_state is None or best_checkpoint_path is None:
        raise RuntimeError("Training did not produce a valid model checkpoint.")

    model.load_state_dict(best_state["model_state"])

    test_metrics = (
        _run_epoch(
            model,
            test_loader,
            criterion,
            device,
            optimizer=None,
            scaler=None,
            autocast_enabled=amp_enabled,
            progress_desc="Test",
        )
        if test_loader
        else None
    )
    LOGGER.info("Test metrics: %s", json.dumps(test_metrics, indent=2) if test_metrics else "n/a")
    if test_metrics:
        step = best_state["epoch"]
        tb_writer.add_scalar("loss/test", test_metrics["loss"], step)
        tb_writer.add_scalar("metrics/test_accuracy", test_metrics["accuracy"], step)
        tb_writer.add_scalar("metrics/test_f1", test_metrics["f1"], step)
        tb_writer.flush()

    with open(config.output_dir / "history.json", "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    if test_loader:
        predictions = _evaluate_predictions(model, test_loader, device)
        pd.DataFrame(predictions).to_csv(config.output_dir / "test_predictions.csv", index=False)

    summary = {
        "checkpoint": str(best_checkpoint_path),
        "best_epoch": best_state["epoch"],
        "best_accuracy": best_state["best_accuracy"],
        "best_val_loss": best_state["best_val_loss"],
        "val_metrics": best_state["val_metrics"],
        "train_metrics": best_state["train_metrics"],
        "amp_enabled": amp_enabled,
        "resume_from": str(config.resume_from) if config.resume_from else None,
        "history_path": str(config.output_dir / "history.json"),
        "test_metrics": test_metrics,
        "retained_checkpoints": {
            "epoch": 10,
            "best": 10,
        },
        "tensorboard_logdir": str(tb_dir),
    }
    with open(config.output_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    tb_writer.close()
    return summary


def _gather_image_paths(inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_file():
            paths.append(path)
        elif path.is_dir():
            paths.extend(sorted(p for p in path.glob("**/*.png") if p.is_file()))
        else:
            LOGGER.warning("Skipping missing path: %s", item)
    if not paths:
        raise FileNotFoundError("No images found for inference.")
    return paths


def predict_images(
    checkpoint_path: Path,
    inputs: Sequence[str],
    device_spec: str = "auto",
) -> pd.DataFrame:
    device = _resolve_device(device_spec)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = ModelConfig(**checkpoint["model_config"])
    model = VSDLM(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    normalization = checkpoint["normalization"]
    mean = normalization.get("mean", DEFAULT_MEAN)
    std = normalization.get("std", DEFAULT_STD)
    image_size = normalization.get("image_size", 112)
    _, eval_transform = _build_transforms(image_size, mean, std)

    image_paths = _gather_image_paths(inputs)

    records: List[Dict[str, Any]] = []
    with torch.no_grad():
        for image_path in image_paths:
            pil_image = Image.open(image_path).convert("RGB")
            tensor = eval_transform(pil_image).unsqueeze(0).to(device)
            logits = model(tensor)
            prob = torch.sigmoid(logits)[0].item()
            records.append(
                {
                    "path": str(image_path),
                    "logit": float(logits.item()),
                    "prob_open": float(prob),
                }
            )

    return pd.DataFrame(records)


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    opset: int = 17,
    device_spec: str = "auto",
) -> None:
    device = _resolve_device(device_spec)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = ModelConfig(**checkpoint["model_config"])
    model = VSDLM(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    normalization = checkpoint["normalization"]
    image_size = normalization.get("image_size", 112)

    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    class _ONNXProbWrapper(nn.Module):
        def __init__(self, base_model: nn.Module) -> None:
            super().__init__()
            self.base_model = base_model

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = self.base_model(x)
            return torch.sigmoid(logits)

    export_base = copy.deepcopy(model)
    export_base.to(device)
    export_base.eval()
    export_model = _ONNXProbWrapper(export_base)

    torch.onnx.export(
        export_model,
        dummy,
        output_path,
        input_names=["images"],
        output_names=["prob_open"],
        dynamic_axes={"images": {0: "batch"}, "prob_open": {0: "batch"}},
        opset_version=opset,
    )
    LOGGER.info("Exported ONNX model to %s", output_path)

    try:
        import onnx
        from onnxsim import simplify

        onnx_model = onnx.load(output_path)
        simplified_model, check = simplify(onnx_model)
        if check:
            simplified_model = _remove_batchnorm_from_onnx(simplified_model)
            onnx.save(simplified_model, output_path)
            LOGGER.info("Simplified ONNX model with onnxsim at %s", output_path)
        else:
            LOGGER.warning("onnxsim simplification check failed; keeping original model.")
    except Exception as exc:
        LOGGER.warning("onnxsim simplification failed: %s", exc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VSDLM training and inference pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the VSDLM binary classifier.")
    train_parser.add_argument("--data_root", type=Path, required=True)
    train_parser.add_argument("--output_dir", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--batch_size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--num_workers", type=int, default=4)
    train_parser.add_argument("--image_size", type=int, default=48)
    train_parser.add_argument("--train_ratio", type=float, default=0.8)
    train_parser.add_argument("--val_ratio", type=float, default=0.2)
    train_parser.add_argument("--test_ratio", type=float, default=0.0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--base_channels", type=int, default=32)
    train_parser.add_argument("--num_blocks", type=int, default=4)
    train_parser.add_argument("--dropout", type=float, default=0.3)
    train_parser.add_argument("--device", type=str, default="auto")
    train_parser.add_argument("--resume", type=Path, help="Resume training from a checkpoint file.")
    train_parser.add_argument("--use_amp", action="store_true", help="Enable mixed precision training (CUDA only).")
    train_parser.add_argument("--verbose", action="store_true")

    predict_parser = subparsers.add_parser("predict", help="Run inference with a trained checkpoint.")
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--inputs", nargs="+", required=True, help="Image files or directories.")
    predict_parser.add_argument("--output", type=Path, help="Optional CSV path to save predictions.")
    predict_parser.add_argument("--device", type=str, default="auto")

    onnx_parser = subparsers.add_parser("exportonnx", help="Export a trained checkpoint to ONNX.")
    onnx_parser.add_argument("--checkpoint", type=Path, required=True)
    onnx_parser.add_argument("--output", type=Path, required=True)
    onnx_parser.add_argument("--opset", type=int, default=17)
    onnx_parser.add_argument("--device", type=str, default="auto")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        config = TrainConfig(
            data_root=args.data_root,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            image_size=args.image_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            base_channels=args.base_channels,
            num_blocks=args.num_blocks,
            dropout=args.dropout,
            device=args.device,
            resume_from=args.resume,
            use_amp=args.use_amp,
        )
        train_pipeline(config, verbose=args.verbose)
    elif args.command == "predict":
        df = predict_images(args.checkpoint, args.inputs, device_spec=args.device)
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Saved predictions to {args.output}")
        else:
            print(df.to_string(index=False))
    elif args.command == "exportonnx":
        export_to_onnx(args.checkpoint, args.output, opset=args.opset, device_spec=args.device)
    else:
        parser.error(f"Unknown command: {args.command}")
