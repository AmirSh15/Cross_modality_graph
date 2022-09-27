# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import time
import weakref
from typing import List, Mapping, Optional

import numpy as np
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

import CrossModalGraph.train_utils.comm as comm
from CrossModalGraph.utils.events import EventStorage, get_event_storage
from CrossModalGraph.utils.logger import _log_api_usage

__all__ = ["HookBase", "TrainerBase", "SimpleTrainer", "AMPTrainer"]


class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.

    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:
    ::
        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        iter += 1
        hook.after_train()

    Notes:
        1. In the hook method, users can access ``self.trainer`` to access more
           properties about the context (e.g., model, current iteration, or config
           if using :class:`DefaultTrainer`).

        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.

           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.

    """

    trainer: "TrainerBase" = None
    """
    A weak reference to the trainer object. Set by the trainer when the hook is registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass

    def state_dict(self):
        """
        Hooks are stateless by default, but can be made checkpointable by
        implementing `state_dict` and `load_state_dict`.
        """
        return {}


class TrainerBase:
    """
    Base class for iterative trainer with hooks.

    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.

    Attributes:
        iter(int): the current iteration.

        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.

        max_iter(int): The iteration to end training.

        storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self) -> None:
        self._hooks: List[HookBase] = []
        self.iter: int = 0
        self.start_iter: int = 0
        self.max_iter: int
        self.storage: EventStorage
        _log_api_usage("trainer." + self.__class__.__name__)

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                # self.iter == max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except StopIteration:
                logger.exception("Early Stopping happened.")
                # raise
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        self.storage.iter = self.iter
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        # Maintain the invariant that storage.iter == trainer.iter
        # for the entire execution of each step
        self.storage.iter = self.iter

        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def run_step(self):
        raise NotImplementedError

    def state_dict(self):
        ret = {"iteration": self.iter}
        hooks_state = {}
        for h in self._hooks:
            sd = h.state_dict()
            if sd:
                name = type(h).__qualname__
                if name in hooks_state:
                    # TODO handle repetitive stateful hooks
                    continue
                hooks_state[name] = sd
        if hooks_state:
            ret["hooks"] = hooks_state
        return ret

    def load_state_dict(self, state_dict):
        logger = logging.getLogger(__name__)
        self.iter = state_dict["iteration"]
        for key, value in state_dict.get("hooks", {}).items():
            for h in self._hooks:
                try:
                    name = type(h).__qualname__
                except AttributeError:
                    continue
                if name == key:
                    h.load_state_dict(value)
                    break
            else:
                logger.warning(
                    f"Cannot find the hook '{key}', its state_dict is ignored."
                )


class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization,
    optionally using data-parallelism.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, model, data_loader, optimizer, config=None):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer
        self.accumulative_counter = 0
        self.iters_to_accumulate = config.SOLVER.ITERS_TO_ACCUMULATE
        self.grd_clip = config.SOLVER.MY_CLIP_GRADIENTS.ENABLED
        self.grd_value = config.SOLVER.CLIP_GRADIENTS.CLIP_VALUE

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict, metric = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        # self.optimizer.zero_grad()
        losses.backward()

        # compute gradient norm
        grad_norm = gradient_utils(self.model, self.grd_clip, self.grd_value)

        self._write_metrics(loss_dict, data_time)
        # just support mAP and accuracy, to add go to events line 466
        self._write_metrics(metric, data_time)
        # # write gradient norm
        self._write_metrics(grad_norm, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        if (self.accumulative_counter + 1) % self.iters_to_accumulate == 0:
            self.accumulative_counter = 0

            self.optimizer.step()

            self.optimizer.zero_grad()

        # increment the iteration counter
        self.accumulative_counter += 1

    def _write_metrics(
        self,
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ) -> None:
        SimpleTrainer.write_metrics(loss_dict, data_time, prefix)

    @staticmethod
    def write_metrics(
        loss_dict: Mapping[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
    ) -> None:
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix for logging keys
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={storage.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            for k, v in metrics_dict.items():
                if "loss" in k:
                    storage.put_scalar(
                        "{}total_loss".format(prefix), total_losses_reduced
                    )
                    break
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict["optimizer"])


class AMPTrainer(SimpleTrainer):
    """
    Like :class:`SimpleTrainer`, but uses PyTorch's native automatic mixed precision
    in the training loop.
    """

    def __init__(self, model, data_loader, optimizer, grad_scaler=None, config=None):
        """
        Args:
            model, data_loader, optimizer: same as in :class:`SimpleTrainer`.
            grad_scaler: torch GradScaler to automatically scale gradients.
        """
        unsupported = (
            "AMPTrainer does not support single-process multi-device training!"
        )
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        super().__init__(model, data_loader, optimizer, config)

        if grad_scaler is None:
            from torch.cuda.amp import GradScaler

            grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler
        self.accumulative_counter = config.SOLVER.ITERS_TO_ACCUMULATE
        self.grd_clip = config.SOLVER.MY_CLIP_GRADIENTS.ENABLED
        self.grd_value = config.SOLVER.CLIP_GRADIENTS.CLIP_VALUE

    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert (
            torch.cuda.is_available()
        ), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        with autocast():
            loss_dict, metric = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        # self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        # compute gradient norm
        grad_norm = gradient_utils(self.model, self.grd_clip, self.grd_value)

        self._write_metrics(loss_dict, data_time)
        self._write_metrics(metric, data_time)
        self._write_metrics(grad_norm, data_time)

        # self.grad_scaler.step(self.optimizer)
        # self.grad_scaler.update()

        if (self.accumulative_counter + 1) % self.iters_to_accumulate == 0:
            self.accumulative_counter = 0

            # Unscales the gradients of optimizer's assigned params in-place
            self.grad_scaler.unscale_(self.optimizer)

            # Unscales gradients and calls
            # or skips optimizer.step()
            self.grad_scaler.step(self.optimizer)

            # Updates the scale for next iteration
            self.grad_scaler.update()

            self.optimizer.zero_grad()

        # increment the iteration counter
        self.accumulative_counter += 1

    def state_dict(self):
        ret = super().state_dict()
        ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.grad_scaler.load_state_dict(state_dict["grad_scaler"])


def compute_gradient_norm(parameters, norm_type=2):
    """
    Compute gradient norm of a list of parameters.
    This is a modified version of torch.nn.utils.clip_grad_norm_ that
    operates on a list of parameters, rather than a single Variable.
    """
    parameters = [p for p in parameters if p.grad is not None and p.requires_grad]
    if len(parameters) == 0:
        total_norm = 0.0
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )

    # parameters = list(filter(lambda p: p.grad is not None, parameters))
    # total_norm = 0
    # for p in parameters:
    #     param_norm = p.grad.data.norm(norm_type)
    #     total_norm += param_norm.item() ** norm_type
    # total_norm = total_norm ** (1.0 / norm_type)

    return total_norm


def gradient_utils(model, grd_clip: bool, grd_value: float):
    """
    Compute gradient norm of submodules separately and total.

    Args:
        model: model to compute gradient norm
        grd_clip: gradient clipping (boolean)
        grd_value: gradient clipping value
    """
    # apply gradient clipping
    if grd_clip:
        # compute gradient norm
        audio_head_grad_norm = compute_gradient_norm(model.audio_head.parameters())
        video_head_grad_norm = compute_gradient_norm(model.video_head.parameters())
        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grd_value)

        # compute gradient norm
        clipped_total_grad_norm = compute_gradient_norm(model.parameters())
        clipped_audio_head_grad_norm = compute_gradient_norm(
            model.audio_head.parameters()
        )
        clipped_video_head_grad_norm = compute_gradient_norm(
            model.video_head.parameters()
        )
    else:
        audio_head_grad_norm = clipped_total_grad_norm = compute_gradient_norm(
            model.audio_head.parameters()
        )
        video_head_grad_norm = clipped_audio_head_grad_norm = compute_gradient_norm(
            model.video_head.parameters()
        )
        total_grad_norm = clipped_video_head_grad_norm = compute_gradient_norm(
            model.parameters()
        )

    return {
        "audio_head_grad_norm": audio_head_grad_norm
        if not (
            np.isnan(audio_head_grad_norm.item())
            and np.isinf(audio_head_grad_norm.item())
        )
        else torch.tensor(0.0),
        "video_head_grad_norm": video_head_grad_norm
        if not (
            np.isnan(video_head_grad_norm.item())
            and np.isinf(video_head_grad_norm.item())
        )
        else torch.tensor(0.0),
        "total_grad_norm": total_grad_norm
        if not (np.isnan(total_grad_norm.item()) and np.isinf(total_grad_norm.item()))
        else torch.tensor(0.0),
        "clipped_total_grad_norm": clipped_total_grad_norm
        if not (
            np.isnan(clipped_total_grad_norm.item())
            and np.isinf(clipped_total_grad_norm.item())
        )
        else torch.tensor(0.0),
        "clipped_audio_head_grad_norm": clipped_audio_head_grad_norm
        if not (
            np.isnan(clipped_audio_head_grad_norm.item())
            and np.isinf(clipped_audio_head_grad_norm.item())
        )
        else torch.tensor(0.0),
        "clipped_video_head_grad_norm": clipped_video_head_grad_norm
        if not (
            np.isnan(clipped_video_head_grad_norm.item())
            and np.isinf(clipped_video_head_grad_norm.item())
        )
        else torch.tensor(0.0),
    }
