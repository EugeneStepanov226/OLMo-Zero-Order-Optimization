"""Run this script with 'torchrun'."""

import gzip
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Optional, TextIO

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP

from olmo.config import (
    CheckpointType,
    DDPGradSyncMode,
    DistributedStrategy,
    TrainConfig,
)
from olmo.data import build_train_dataloader
from olmo.eval import build_evaluators
from olmo.exceptions import OLMoCliError, OLMoConfigurationError
from olmo.model import OLMo
from olmo.optim import BoltOnWarmupScheduler, build_optimizer, build_scheduler
from olmo.torch_util import (
    SingleAccelerator,
    barrier,
    get_default_device,
    get_global_rank,
    get_local_rank,
    get_local_world_size,
    get_world_size,
    is_distributed,
    move_to_device,
    peak_gpu_memory,
    seed_all,
)
from olmo.train import Trainer
from olmo.util import (
    add_cached_path_clients,
    clean_opt,
    find_latest_checkpoint,
    log_extra_field,
    prepare_cli_environment,
)

log = logging.getLogger("train")


# Parameter-name prefixes that identify the token/positional embeddings and the
# LM head. We split these out so they train with first-order AdamW while the
# rest of the network trains with the configured zero-order optimizer.
_HEAD_PREFIXES = ("transformer.ff_out.",)
_EMB_PREFIXES = ("transformer.wte.", "transformer.wpe.")


def _is_emb_head_param(name: str, include_embeddings: bool = True) -> bool:
    if name.startswith("module."):
        name = name[len("module.") :]
    prefixes = _HEAD_PREFIXES + (_EMB_PREFIXES if include_embeddings else ())
    return any(name.startswith(p) for p in prefixes)


class _MixedFOZOTrainer(Trainer):
    """Trainer that runs an AdamW step on embedding+head before the regular ZO step.

    The FO step is memory-frugal: it computes gradients *only* for the embedding/head
    tensors via ``torch.autograd.grad`` (so no gradient buffers are allocated for the
    ~1B body parameters that ZO handles), running the forward through the *unwrapped*
    model so DDP's reducer is never armed. Cross-rank averaging is done manually.
    """

    fo_optim: Optional[torch.optim.Optimizer] = None  # attached after construction

    @property
    def _fo_base_lr(self) -> float:
        fo_lr = self.cfg.optimizer.fo_learning_rate
        return fo_lr if fo_lr is not None else self.cfg.optimizer.learning_rate

    def _train_step_zero_order(self, batch, reduce_global_loss: bool = True):
        assert self.fo_optim is not None, "fo_optim must be attached before training"

        # ---- First-order step on embedding + head only ----
        self.fo_optim.zero_grad(set_to_none=True)
        batch = move_to_device(batch, self.device)

        fo_params = [p for group in self.fo_optim.param_groups for p in group["params"]]
        micro_batches = self.split_batch(batch)
        batch_size_in_tokens = batch["input_ids"].numel()
        autocast_device = "mps" if self.device.type == "mps" else "cuda"

        accum_grads = [torch.zeros_like(p, dtype=torch.float32) for p in fo_params]
        for micro_batch in micro_batches:
            with torch.autocast(autocast_device, enabled=True, dtype=self.cfg.autocast_precision):
                # Forward through the UNWRAPPED model to avoid arming the DDP reducer.
                logits = self.model(
                    input_ids=micro_batch["input_ids"],
                    attention_mask=micro_batch.get("attention_mask"),
                    attention_bias=micro_batch.get("attention_bias"),
                    doc_lens=micro_batch.get("doc_lens"),
                    max_doc_lens=micro_batch.get("max_doc_lens"),
                ).logits
                logits_for_loss = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
                labels = self.get_labels(micro_batch).view(-1)
                ce_loss, z_loss = self.loss_fn(
                    logits_for_loss,
                    labels,
                    ignore_index=-100,
                    reduction="sum",
                    compute_z_loss=self.cfg.softmax_auxiliary_loss,
                )
                loss = ce_loss / batch_size_in_tokens
                if z_loss is not None:
                    loss = loss + z_loss / batch_size_in_tokens
            # Gradients ONLY for emb/head — body grads are never materialized.
            grads = torch.autograd.grad(loss, fo_params, retain_graph=False, allow_unused=True)
            for acc, g in zip(accum_grads, grads):
                if g is not None:
                    acc.add_(g.float())

        # Manual cross-rank averaging (DDP reducer was bypassed above).
        if is_distributed():
            for acc in accum_grads:
                dist.all_reduce(acc, op=dist.ReduceOp.SUM)
                acc.div_(get_world_size())

        for p, acc in zip(fo_params, accum_grads):
            p.grad = acc.to(p.dtype)

        if self.cfg.max_grad_norm is not None and self.cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(fo_params, self.cfg.max_grad_norm)

        fo_lr = self.scheduler.get_lr(self._fo_base_lr, self.scheduler_current, self.scheduler_max)
        for group in self.fo_optim.param_groups:
            group["lr"] = fo_lr
        self.fo_optim.step()

        # ---- Zero-order step on the remaining parameters (unchanged behavior) ----
        return super()._train_step_zero_order(batch, reduce_global_loss=reduce_global_loss)


def main(cfg: TrainConfig) -> None:
    # Ensure run name set.
    if cfg.run_name is None:
        raise OLMoConfigurationError("--run_name is required")
    log_extra_field("run_name", cfg.run_name)

    # Sanity check
    if (cfg.reset_optimizer_state or cfg.reset_trainer_state) and cfg.load_path is None:
        log.warning(
            "You want to reset the optimizer or trainer state, but we're not loading from the checkpoint. The"
            "setting has no effect."
        )

    barrier()

    # Set CUDA device.
    if torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{get_local_rank()}")
        torch.cuda.empty_cache()
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Fill some configuration options.
    cfg.model.precision = cfg.precision
    cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
    assert cfg.device_train_batch_size is not None  # for mypy
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size
    if cfg.optimizer.no_decay_norm_and_bias is not None:
        log.warning(
            "You set the deprecated config option `no_decay_norm_and_bias`. For compatibility, this"
            "setting will take precedence over all other weight decay configurations. Please change"
            "your config to use `decay_norm_and_bias` and `decay_embeddings` instead."
        )
        cfg.optimizer.decay_norm_and_bias = not cfg.optimizer.no_decay_norm_and_bias
        cfg.optimizer.decay_embeddings = not cfg.optimizer.no_decay_norm_and_bias
        cfg.optimizer.no_decay_norm_and_bias = None  # So nobody uses this by accident.

    # Display and save configuration.
    if get_global_rank() == 0:
        if cfg.data.paths is not None and len(cfg.data.paths) < 50:
            log.info("Configuration:")
            log.info(cfg)
        if not cfg.dry_run and (cfg.load_path is None or Path(cfg.load_path).parent != Path(cfg.save_folder)):
            # Save config.
            save_path = Path(cfg.save_folder) / "config.yaml"
            if save_path.is_file() and not cfg.save_overwrite:
                raise OLMoConfigurationError(f"{save_path} already exists, use --save_overwrite to overwrite")
            else:
                log.info(f"Saving config to {save_path}")
                save_path.parent.mkdir(exist_ok=True, parents=True)
                cfg.save(save_path)
            del save_path

    barrier()

    # Maybe start W&B run.
    if cfg.wandb is not None and (get_global_rank() == 0 or not cfg.wandb.rank_zero_only):
        wandb_dir = Path(cfg.save_folder) / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(
            dir=str(wandb_dir),
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags,
            config=cfg.asdict(exclude=["wandb"]),
        )

    barrier()

    # Set seed.
    seed_all(cfg.seed)

    # Construct data loader.
    train_loader = build_train_dataloader(cfg)

    # Construct evaluators.
    evaluators = build_evaluators(cfg, device)
    barrier()

    # Initialize the model.
    log.info("Building model...")
    olmo_model = OLMo(cfg.model)
    log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
    log.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")
    log.info(f"Peak GPU Memory (MB) before {cfg.distributed_strategy}: {int(peak_gpu_memory() or 0)}")

    # Compile one block at a time.
    if cfg.compile is not None:
        if cfg.model.block_group_size != 1:
            raise OLMoConfigurationError("Compile is only supported with block_group_size 1.")
        for block in olmo_model.transformer.blocks:
            block.compile(**cfg.compile.asdict())

    olmo_model.set_activation_checkpointing(cfg.activation_checkpointing)

    if cfg.distributed_strategy == DistributedStrategy.ddp:
        log.info("Wrapping model with DDP...")
        assert cfg.ddp is not None, "DistributedStrategy ddp needs cfg.ddp to be set!"

        if cfg.model.init_device != "cuda":
            raise OLMoConfigurationError("DDP does not work with init_device set to anything other than `cuda`.")

        if cfg.ddp.find_unused_params is True and cfg.ddp.grad_sync_mode != DDPGradSyncMode.micro_batch:
            raise OLMoConfigurationError(
                "`find_unused_params` is set to True. DDP needs to synchronize gradients for every micro-batch to avoid errors. Set `grad_sync_mode` to `micro_batch`."
            )

        param_init_fn = None

        # move to cuda before calling ddp
        dist_model = DDP(olmo_model.to(device), find_unused_parameters=cfg.ddp.find_unused_params)
    elif cfg.distributed_strategy == DistributedStrategy.fsdp:
        # Wrap the model in FSDP.
        log.info("Wrapping model with FSDP...")
        assert cfg.fsdp is not None, "DistributedStrategy fsdp needs cfg.fsdp to be set!"
        wrap_policy = olmo_model.get_fsdp_wrap_policy(cfg.fsdp.wrapping_strategy)

        if version.parse(torch.__version__) >= version.parse("2.1.0"):
            # This prevents any parameters from being initialized twice
            def dummy_init_fn(module: torch.nn.Module) -> None:
                module.to_empty(device=get_default_device())

            param_init_fn = dummy_init_fn
        else:
            param_init_fn = None

        # Set up device mesh for hybrid sharding in order to specify which nodes are assoicated to a given model replica
        device_mesh = None
        hybrid_sharding_fsdp_kwargs = {}
        if cfg.fsdp.sharding_strategy in (ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2):
            if version.parse(torch.__version__) < version.parse("2.2.0"):
                # Device mesh was not added to PyTorch until v2.2.0
                raise OLMoConfigurationError(
                    "OLMo training does not correctly support hybrid sharding before torch 2.2.0"
                )

            from torch.distributed.device_mesh import init_device_mesh

            num_model_replicas = cfg.fsdp.hybrid_sharding_num_model_replicas or (
                get_world_size() // get_local_world_size()
            )

            if num_model_replicas <= 0:
                raise OLMoConfigurationError("fsdp.hybrid_sharding_num_model_replicas must be a positive integer")

            if get_world_size() % num_model_replicas != 0:
                raise OLMoConfigurationError("fsdp.hybrid_sharding_num_model_replicas must divide world size")

            device_mesh = init_device_mesh("cuda", (num_model_replicas, get_world_size() // num_model_replicas))
            hybrid_sharding_fsdp_kwargs["device_mesh"] = device_mesh

        dist_model = FSDP(
            olmo_model,
            sharding_strategy=cfg.fsdp.sharding_strategy,
            mixed_precision=cfg.fsdp_precision,
            auto_wrap_policy=wrap_policy,
            use_orig_params=cfg.fsdp.use_orig_params,  # needed for compile and some of our optimizer/parameter metrics
            limit_all_gathers=True,
            device_id=get_local_rank(),
            param_init_fn=param_init_fn,
            **hybrid_sharding_fsdp_kwargs,
        )
    elif cfg.distributed_strategy == DistributedStrategy.single:
        param_init_fn = None
        if olmo_model is None:
            raise OLMoConfigurationError("Model initialization failed.")
        olmo_model = olmo_model.to(device)
        dist_model = SingleAccelerator(olmo_model)

    # when param_init_fn is None, FSDP will call reset_parameters() automatically
    if param_init_fn is not None or cfg.distributed_strategy == DistributedStrategy.ddp:
        olmo_model.reset_parameters()

    log.info(f"Peak GPU Memory (MB) after {cfg.distributed_strategy}: {int(peak_gpu_memory() or 0)}")
    log.info("Model:")
    log.info(dist_model)

    # Split parameters: LM head (ff_out) and, when fo_include_embeddings is set, the
    # embeddings (wte/wpe) train with FO (AdamW); the rest trains with ZO. We temporarily
    # flip requires_grad off on the FO tensors so `build_optimizer` -> `get_param_groups`
    # skips them.
    fo_include_embeddings = cfg.optimizer.fo_include_embeddings
    emb_head_params = [
        p for name, p in olmo_model.named_parameters() if _is_emb_head_param(name, fo_include_embeddings)
    ]
    for p in emb_head_params:
        p.requires_grad_(False)

    # Construct optimizer and learning rate scheduler.
    optim = build_optimizer(cfg, dist_model)

    # Restore requires_grad and build the first-order optimizer on emb+head.
    for p in emb_head_params:
        p.requires_grad_(True)

    # Split emb vs head so weight decay respects `decay_embeddings` (embeddings get wd=0
    # when decay_embeddings is False, matching get_param_groups for the ZO side).
    emb_fo_params, head_fo_params = [], []
    for name, p in olmo_model.named_parameters():
        if not _is_emb_head_param(name, fo_include_embeddings):
            continue
        base = name[len("module.") :] if name.startswith("module.") else name
        if base.startswith("transformer.ff_out."):
            head_fo_params.append(p)
        else:
            emb_fo_params.append(p)

    emb_weight_decay = cfg.optimizer.weight_decay if cfg.optimizer.decay_embeddings else 0.0
    fo_peak_lr = (
        cfg.optimizer.fo_learning_rate
        if cfg.optimizer.fo_learning_rate is not None
        else cfg.optimizer.learning_rate
    )
    fo_optim = torch.optim.AdamW(
        [
            {"params": head_fo_params, "weight_decay": cfg.optimizer.weight_decay},
            {"params": emb_fo_params, "weight_decay": emb_weight_decay},
        ],
        lr=fo_peak_lr,
        betas=tuple(cfg.optimizer.betas),
        eps=cfg.optimizer.eps,
    )
    log.info(
        f"FO (AdamW) optimizer: {len(head_fo_params)} head + {len(emb_fo_params)} embedding tensors, "
        f"peak_lr={fo_peak_lr}; ZO optimizer covers the remaining parameters."
    )

    scheduler = build_scheduler(cfg)

    # Data indices file.
    indices_file: Optional[TextIO] = None
    if cfg.save_data_indices:
        indices_file_path = Path(cfg.save_folder) / f"data-indices/rank{get_global_rank()}.tsv.gz"
        if indices_file_path.exists() and not cfg.save_overwrite:
            raise OLMoConfigurationError(f"{indices_file_path} already exists, use --save_overwrite to overwrite")
        indices_file_path.parent.mkdir(exist_ok=True, parents=True)
        indices_file = gzip.open(indices_file_path, "wt")

    # Consolidate components into `Trainer` object.
    with _MixedFOZOTrainer(
        cfg=cfg,
        epoch=cfg.epoch,
        model=olmo_model,
        dist_model=dist_model,
        optim=optim,
        scheduler=scheduler,
        train_loader=train_loader,
        device=device,
        evaluators=evaluators,
        indices_file=indices_file,
    ) as trainer:
        trainer.fo_optim = fo_optim
        if cfg.try_load_latest_save:
            checkpoint_dir = None
            if (
                cfg.save_folder is not None
                and (checkpoint_dir := find_latest_checkpoint(cfg.save_folder)) is not None
            ):
                log.info("Setting load path to local checkpoint %s", checkpoint_dir)
                cfg.load_path = str(checkpoint_dir)
            elif (
                cfg.remote_save_folder is not None
                and (checkpoint_dir := find_latest_checkpoint(cfg.remote_save_folder)) is not None
            ):
                log.info("Setting load path to remote checkpoint %s", checkpoint_dir)
                cfg.load_path = str(checkpoint_dir)
            if checkpoint_dir is not None and not cfg.restore_dataloader:
                log.info(
                    "You set restore_dataloader=False, but try_load_latest_save=True. If we were to run like "
                    "this, it would overwrite your previous checkpoints. I will assume you didn't mean that, "
                    "and set restore_dataloader=True."
                )
                cfg.restore_dataloader = True
            if checkpoint_dir is not None and cfg.reset_trainer_state:
                log.info(
                    "You set both reset_trainer_state=True, and try_load_latest_save=True. If we were to "
                    "run like this, it would reset your trainer state right now even though we're in the "
                    "middle of a run. I will assume you didn't mean that, and set "
                    "reset_trainer_state=False."
                )
                cfg.reset_trainer_state = False
            if checkpoint_dir is not None and cfg.reset_optimizer_state:
                log.info(
                    "You set both reset_optimizer_state=True, and try_load_latest_save=True. If we were to "
                    "run like this, it would reset your optimizer state right now even though we're in the "
                    "middle of a run. I will assume you didn't mean that, and set "
                    "reset_optimizer_state=False."
                )
                cfg.reset_optimizer_state = False

        if not cfg.dry_run and not cfg.no_pre_train_checkpoint and cfg.load_path is None:
            if cfg.distributed_strategy == DistributedStrategy.ddp:
                checkpoint_type = CheckpointType.unsharded

                if cfg.save_interval_unsharded is None:
                    log.warning(
                        "DDP requires setting `save_interval_unsharded`. Using the value set for `save_interval`."
                    )
                    cfg.save_interval_unsharded = cfg.save_interval

                if cfg.save_num_unsharded_checkpoints_to_keep == 0:
                    log.warning(
                        "DDP requires setting `save_num_unsharded_checkpoints_to_keep`. Using the value set for `save_num_checkpoints_to_keep`."
                    )
                    cfg.save_num_unsharded_checkpoints_to_keep = cfg.save_num_checkpoints_to_keep
            elif cfg.distributed_strategy == DistributedStrategy.fsdp:
                checkpoint_type = (
                    CheckpointType.sharded if cfg.save_num_checkpoints_to_keep != 0 else CheckpointType.unsharded
                )
            elif cfg.distributed_strategy == DistributedStrategy.single:
                checkpoint_type = CheckpointType.unsharded

                if cfg.save_interval_unsharded is None:
                    log.warning(
                        "single accelerator training requires setting `save_interval_unsharded`. Using the value set for `save_interval`."
                    )
                    cfg.save_interval_unsharded = cfg.save_interval

                if cfg.save_num_unsharded_checkpoints_to_keep == 0:
                    log.warning(
                        "single accelerator training requires setting `save_num_unsharded_checkpoints_to_keep`. Using the value set for `save_num_checkpoints_to_keep`."
                    )
                    cfg.save_num_unsharded_checkpoints_to_keep = cfg.save_num_checkpoints_to_keep

            # We save a checkpoint up-front to make sure this won't fail (due to disk space or whatever).
            log.info("Saving pre-train checkpoint...")
            checkpoint_path, local_checkpoint_cache = trainer.save_checkpoint(checkpoint_type=checkpoint_type)
            log.info(f"Checkpoint saved to {checkpoint_path}")

            # And they we verify that we can load it.
            log.info("Attempting to load pre-train checkpoint...")
            trainer.restore_checkpoint(
                checkpoint_path, checkpoint_type=checkpoint_type, local_cache=local_checkpoint_cache
            )
            log.info("Checkpoint successfully loaded")

            # NOTE: https://github.com/allenai/LLM/issues/233
            #  log.info("Removing pre-train checkpoint...")
            #  trainer.remove_checkpoint(checkpoint_type=checkpoint_type)
            #  log.info("Successfully removed checkpoint")

        if cfg.load_path is not None:
            log.info(f"Loading checkpoint from {cfg.load_path}...")
            trainer.restore_checkpoint(
                cfg.load_path,
                load_optimizer_state=not cfg.reset_optimizer_state,
                load_trainer_state=not cfg.reset_trainer_state,
                sharded_checkpointer=cfg.load_path_sharded_checkpointer,
            )
            log.info("Checkpoint successfully loaded")

            # If we have to, set a new scheduler:
            if cfg.reset_optimizer_state and not cfg.reset_trainer_state:
                trainer.scheduler = BoltOnWarmupScheduler.wrap(
                    trainer.scheduler,
                    trainer.global_step,
                    int(trainer.global_step + cfg.scheduler.t_warmup),
                )

        if cfg.force_save_unsharded and cfg.distributed_strategy != DistributedStrategy.ddp:
            log.info("Saving unsharded checkpoint...")
            checkpoint_path, _ = trainer.save_checkpoint(checkpoint_type=CheckpointType.unsharded)
            log.info(f"Unsharded checkpoint saved to {checkpoint_path}")

        if not cfg.dry_run:
            log.info("Starting training...")
            trainer.fit()
            log.info("Training complete")
        else:
            log.info("Dry run complete")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")
    log.info(f"Multiprocessing start method set to '{mp.get_start_method()}'")
    if torch.cuda.is_available():
        # Set CUDA device.
        torch.cuda.set_device(f"cuda:{get_local_rank()}")

        # Initialize process group.
        device_as_string = f"cuda:{get_local_rank()}"
        torch.cuda.set_device(
            device_as_string
        )  # Set this early to prevent GPU 0 from picking up a bunch of tensors it shouldn't have.
        dist.init_process_group(
            backend="nccl", timeout=timedelta(minutes=30), device_id=torch.device(device_as_string)
        )
    elif torch.backends.mps.is_available():
        if not os.getenv("RANK"):
            os.environ["RANK"] = "0"
        if not os.getenv("WORLD_SIZE"):
            os.environ["WORLD_SIZE"] = "1"
        if not os.getenv("MASTER_ADDR"):
            os.environ["MASTER_ADDR"] = "0.0.0.0"
        if not os.getenv("MASTER_PORT"):
            os.environ["MASTER_PORT"] = "24501"
        dist.init_process_group(backend="gloo", timeout=timedelta(minutes=30))

    else:
        dist.init_process_group(backend="gloo", timeout=timedelta(minutes=30))

    log.info("Process group initialized")

    prepare_cli_environment()
    log.info("CLI environment prepared")

    add_cached_path_clients()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OLMoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    if torch.backends.mps.is_available():
        log.info("Device is MPS. Updating config...")
        cfg.model.init_device = "mps"
        cfg.distributed_strategy = "single"  # type: ignore

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        log.info("Device is CPU. Updating config...")
        cfg.model.init_device = "cpu"
        cfg.distributed_strategy = "single"  # type: ignore
    main(cfg)
