"""
Trainer for retrieval training and validation. Holds the main training loop.
"""

import json
import logging
import os
from collections import defaultdict
from collections.abc import Mapping
from glob import glob
from pathlib import Path
from timeit import default_timer as timer
from typing import Dict, List, Optional, Tuple, Union
from typing import ClassVar
import copy

import numpy as np
import torch as th
from torch import nn
from torch.cuda.amp import autocast
from torch.utils import data
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from coot.configs_retrieval import ExperimentTypesConst
from mart.caption_eval_tools import get_reference_files
from mart.configs_mart import MartConfig, MartMetersConst as MMeters
from mart.evaluate_language import evaluate_language_files
from mart.evaluate_repetition import evaluate_repetition_files
from mart.evaluate_stats import evaluate_stats_files
from mart.optimization import BertAdam, EMA
from mart.recursive_caption_dataset import RecursiveCaptionDataset, prepare_batch_inputs
from mart.translator import Translator
from nntrainer import trainer_base
from nntrainer.experiment_organization import ExperimentFilesHandler
from nntrainer.metric import TRANSLATION_METRICS, TextMetricsConst, TextMetricsConstEvalCap
from nntrainer.models import BaseModelManager
from nntrainer.trainer_configs import BaseTrainerState
from nntrainer.utils import TrainerPathConst


def cal_performance(pred, gold):
    pred = pred.max(2)[1].contiguous().view(-1)
    gold = gold.contiguous().view(-1)
    valid_label_mask = gold.ne(RecursiveCaptionDataset.IGNORE)
    pred_correct_mask = pred.eq(gold)
    n_correct = pred_correct_mask.masked_select(valid_label_mask).sum().item()
    return n_correct


# only log the important ones to console
TRANSLATION_METRICS_LOG = ["Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "re4"]


class MartFilesHandler(ExperimentFilesHandler):
    """
    Overwrite default filehandler to add some more paths.
    """

    def __init__(self, exp_group: str, exp_name: str, run_name: str, log_dir: str = TrainerPathConst.DIR_EXPERIMENTS,
                 annotations_dir: str = TrainerPathConst.DIR_ANNOTATIONS):
        super().__init__(ExperimentTypesConst.CAPTION, exp_group, exp_name, run_name, log_dir=log_dir)
        self.annotations_dir = annotations_dir
        self.path_caption = self.path_base / TrainerPathConst.DIR_CAPTION

    def get_translation_files(self, epoch: Union[int, str], split: str) -> Path:
        """
        Get all file paths for storing translation results and evaluation.

        Args:
            epoch: Epoch.
            split: dataset split (val, test)

        Returns:
            Path to store raw model output and ground truth.
        """
        return self.path_caption / f"{TrainerPathConst.FILE_PREFIX_TRANSL_RAW}_{epoch}_{split}.json"

    def setup_dirs(self, *, reset: bool = False) -> None:
        """
        Call super class to setup directories and additionally create the caption folder.

        Args:
            reset:

        Returns:
        """
        super().setup_dirs(reset=reset)
        os.makedirs(self.path_caption, exist_ok=True)


class MartModelManager(BaseModelManager):
    """
    Wrapper for MART models.
    """

    def __init__(self, cfg: MartConfig, model: nn.Module):
        super().__init__(cfg)
        # update config type hints
        self.cfg: MartConfig = self.cfg
        self.model_dict: [str, nn.Module] = {"model": model}


class MartTrainerState(BaseTrainerState):
    prev_best_score: ClassVar[float] = 0.0
    es_cnt: ClassVar[int] = 0


class MartTrainer(trainer_base.BaseTrainer):
    """
    Trainer for retrieval.

    Notes:
        The parent TrainerBase takes care of all the basic stuff: Setting up directories and logging,
        determining device and moving models to cuda, setting up checkpoint loading and metrics.

    Args:
        cfg: Loaded configuration instance.
        model: Model.
        exp_group: Experiment group.
        exp_name: Experiment name.
        run_name: Experiment run.
        train_loader_length: Length of the train loader, required for some LR schedulers.
        log_dir: Directory to put results.
        log_level: Log level. None will default to INFO = 20 if a new logger is created.
        logger: Logger. With the default None, it will be created by the trainer.
        print_graph: Print graph and forward pass of the model.
        reset: Delete entire experiment and restart from scratch.
        load_best: Whether to load the best epoch (default loads last epoch to continue training).
        load_epoch: Whether to load a specific epoch.
        load_model: Load model given by file path.
        inference_only: Removes some parts that are not needed during inference for speedup.
        annotations_dir: Folder with ground truth captions.
    """

    def __init__(
            self, cfg: MartConfig, model: nn.Module,
            exp_group: str, exp_name: str, run_name: str, train_loader_length: int, *,
            log_dir: str = "experiments", log_level: Optional[int] = None,
            logger: Optional[logging.Logger] = None, print_graph: bool = False, reset: bool = False,
            load_best: bool = False, load_epoch: Optional[int] = None, load_model: Optional[str] = None,
            inference_only: bool = False, annotations_dir: str = TrainerPathConst.DIR_ANNOTATIONS):
        # create a wrapper for the model
        model_mgr = MartModelManager(cfg, model)

        # overwrite default experiment files handler
        exp = MartFilesHandler(exp_group, exp_name, run_name, log_dir=log_dir, annotations_dir=annotations_dir)
        exp.setup_dirs(reset=reset)

        super().__init__(
            cfg, model_mgr, exp_group, exp_name, run_name, train_loader_length, ExperimentTypesConst.CAPTION,
            log_dir=log_dir, log_level=log_level, logger=logger, print_graph=print_graph, reset=reset,
            load_best=load_best, load_epoch=load_epoch, load_model=load_model, is_test=inference_only,
            exp_files_handler=exp)
        self.model = model
        # ---------- setup ----------

        # update type hints from base classes to inherited classes
        self.cfg: MartConfig = self.cfg
        self.model_mgr: MartModelManager = self.model_mgr
        self.exp: MartFilesHandler = self.exp

        self.KEYWORDS =  [
            "kill", "killed", "picks up",
            "baron", "dragon", "drake", "elder",
            "tower", "turret", "inhibitor",
        ]

        # # overwrite default state with inherited trainer state in case we need additional state fields
        # self.state = RetrievalTrainerState()

        # ---------- loss ----------

        # loss is created directly in the mart model and not needed here

        # ---------- additional metrics ----------
        # train loss and accuracy
        self.metrics.add_meter(MMeters.TRAIN_LOSS_PER_WORD, use_avg=False)
        self.metrics.add_meter(MMeters.TRAIN_ACC, use_avg=False)
        self.metrics.add_meter(MMeters.VAL_LOSS_PER_WORD, use_avg=False)
        self.metrics.add_meter(MMeters.VAL_ACC, use_avg=False)

        # track gradient clipping manually
        self.metrics.add_meter(MMeters.GRAD, per_step=True, reset_avg_each_epoch=True)

        # translation metrics (bleu etc.)
        for meter_name in TRANSLATION_METRICS.values():
            self.metrics.add_meter(meter_name, use_avg=False)

        # ---------- optimization ----------

        self.optimizer = None
        self.lr_scheduler = None
        self.ema = EMA(cfg.ema_decay)
        # skip optimizer if not training
        if not self.is_test:
            # Prepare optimizer
            param_optimizer = list(model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 "weight_decay": 0.01},
                {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
            ]
            if cfg.ema_decay > 0:
                # register EMA params
                self.logger.info(f"Registering {sum(p.numel() for p in model.parameters())} params for EMA")
                all_names = []
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        self.ema.register(name, p.data)
                    all_names.append(name)
                self.logger.debug('\n'.join(all_names))

            num_train_optimization_steps = train_loader_length * cfg.train.num_epochs
            self.optimizer = BertAdam(optimizer_grouped_parameters, lr=cfg.lr, warmup=cfg.lr_warmup_proportion,
                                      t_total=num_train_optimization_steps, e=cfg.eps,
                                      schedule="warmup_linear")

        # ---------- Translator ----------

        self.translator = Translator(self.model, self.cfg, logger=self.logger)

        # post init hook for checkpoint loading
        self.hook_post_init()

        if self.load and not self.load_model:
            # reload EMA weights from checkpoint (the shadow) and save the model parameters (the original)
            ema_file = self.exp.get_models_file_ema(self.load_ep)
            self.logger.info(f"Update EMA from {ema_file}")
            self.ema.set_state_dict(th.load(str(ema_file)))
            self.ema.assign(self.model, update_model=False)

        # disable ema when loading model directly or when decay is 0 / -1
        if self.load_model or cfg.ema_decay <= 0:
            self.ema = None
    def contains_keyword(self, text):
        text = text.lower()
        return any(k in text for k in self.KEYWORDS)
    
    def build_importance_labels(self, input_labels_list, gt_sentences_list):
        """
        input_labels_list: [(N, L)] * step_size
        gt_sentences_list: length = step_size, list of string (GT captions)

        Return:
            importance_labels_list: [(N, L)] * step_size (float tensor)
        """

        step_size = len(input_labels_list)
        N, L = input_labels_list[0].shape
        device = input_labels_list[0].device

        importance_labels_list = []

        # 미래 이벤트 기반 importance 설정
        for t in range(step_size):
            gt_sents = gt_sentences_list[t]
            importance_tensor = th.zeros((N, L), device=device, dtype=th.float32)

            # 현재 setence에서 keyword가 있으면 1, 없으면 0
            for b in range(N):
                s = gt_sents[b]
                if s is not None and self.contains_keyword(s):
                    importance_tensor[b, :] == 1.0
            
            importance_labels_list.append(importance_tensor)

        return importance_labels_list
    
    def train_model(self, train_loader: data.DataLoader, val_loader: data.DataLoader) -> None:
        """
        Train epochs until done.

        Args:
            train_loader: Training dataloader.
            val_loader: Validation dataloader.
        """
        self.hook_pre_train()  # pre-training hook: time book-keeping etc.
        self.steps_per_epoch = len(train_loader)  # save length of epoch

        # ---------- Epoch Loop ----------
        for _epoch in range(self.state.current_epoch, self.cfg.train.num_epochs):
            if self.check_early_stop():
                break
            self.hook_pre_train_epoch()  # pre-epoch hook: set models to train, time book-keeping

            # check exponential moving average
            if self.ema is not None and self.state.current_epoch != 0 and self.cfg.ema_decay != -1:
                # use normal parameters for training, not EMA model
                self.ema.resume(self.model)

            th.autograd.set_detect_anomaly(True)

            total_loss = 0
            n_word_total = 0
            n_word_correct = 0
            accum_steps = 32

            # ---------- Dataloader Iteration (Batch Loop) ----------
            # batch[0]은 collate_fn에서 반환한 'List[List[Dict]]' 입니다.
            # 구조: [ [Video A의 Clip 1, Clip 2...], [Video B의 Clip 1, Clip 2...] ... ]
            for step, batch in enumerate(train_loader):
                print(f"\n[DEBUG] Batch {step} 데이터 로딩 완료! 학습 시작합니다.")
                self.hook_pre_step_timer()  # hook for step timing

                batch_clips_lists = batch[0]
                batch_meta_lists = batch[2]
                batch_size = len(batch_clips_lists)

                # 1. 현재 배치(여러 영상들) 중 가장 긴 영상의 길이(Step 수) 계산
                max_steps = max([len(clips) for clips in batch_clips_lists])

                # 2. 새로운 배치가 시작되었으므로, 모든 슬롯(Lane)의 메모리 초기화
                # (None을 넘기면 모델 내부에서 초기화됨)
                current_memory = None

                # 3. Optimizer 초기화 (영상 전체에 대해 한 번 업데이트하기 위함)
                self.optimizer.zero_grad()

                # ---------- Inner Loop: Time Step (클립) 단위 순차 진행 ----------
                # t = 0 (모든 영상의 첫 번째 클립), t = 1 (두 번째 클립) ... 순서로 진행
                for t in range(max_steps):
                    if t % 10 == 0:
                        print(f"\r  > Step {step} | Clip {t}/{max_steps} Processing...", end="", flush=True)
                    
                    # --- (A) 현재 스텝(t)의 배치 구성 (Dynamic Batching with Padding) ---
                    current_step_inputs = []
                    current_step_gt_sentences = []

                    for b_i in range(batch_size):
                        video_clips = batch_clips_lists[b_i]
                        video_meta = batch_meta_lists[b_i]

                        if t < len(video_clips):
                            # 아직 영상이 진행 중인 경우 -> 실제 데이터 사용
                            current_step_inputs.append(video_clips[t])
                            gt_sent = video_meta[t]["gt_sentence"]
                            current_step_gt_sentences.append(gt_sent)
                        else:
                            dummy = copy.deepcopy(video_clips[-1])
                            dummy['input_labels'][:] = RecursiveCaptionDataset.IGNORE
                            dummy['input_mask'][:] = 0
                            dummy['input_mask'][0] = 1
                            current_step_inputs.append(dummy)
                            # 패딩 구간은 None 처리
                            current_step_gt_sentences.append(None)

                    # 리스트 형태의 입력을 하나의 배치 텐서로 변환
                    # default_collate: List[Dict] -> Dict[Tensor] (Stacked)
                    collated_input = default_collate(current_step_inputs)
                    
                    # GPU 이동
                    batched_data = prepare_batch_inputs(collated_input, use_cuda=self.cfg.use_cuda,
                                                        non_blocking=self.cfg.cuda_non_blocking)

                    # --- (B) 모델 실행 ---
                    with autocast(enabled=self.cfg.fp16_train):
                        if self.cfg.recurrent:
                            # MART 모델은 리스트 형태의 입력을 기대하므로 리스트로 감싸줌
                            # 각 텐서의 shape: [Batch_Size, Seq_Len(words), Dim]
                            input_ids_list = [batched_data["input_ids"]]
                            video_features_list = [batched_data["video_feature"]]
                            input_masks_list = [batched_data["input_mask"]]
                            token_type_ids_list = [batched_data["token_type_ids"]]
                            input_labels_list = [batched_data["input_labels"]]
                            importance_labels_list = self.build_importance_labels(
                            input_labels_list, [current_step_gt_sentences]
                            )

                            if self.cfg.debug and t == 0:
                                self.logger.info(f"Batch Step {step}, Time {t}, Input IDs: {input_ids_list[0].shape}")

                            # [핵심] past_memory 전달 (이전 스텝의 기억 유지)
                            loss, pred_scores_list, new_memory = self.model(
                            input_ids_list, video_features_list, input_masks_list,
                            token_type_ids_list, input_labels_list,
                            importance_labels_list=importance_labels_list, # [추가]
                            past_memory=current_memory 
                        )
                        
                        elif self.cfg.untied or self.cfg.mtrans:
                            # Untied 모델용 로직 (이 경우 메모리 전달 없음)
                            loss, pred_scores = self.model(
                                batched_data["video_feature"], batched_data["video_mask"],
                                batched_data["text_ids"], batched_data["text_mask"],
                                batched_data["text_labels"]
                            )
                            pred_scores_list = [pred_scores]
                            input_labels_list = [batched_data["text_labels"]]
                            new_memory = None
                        
                        else:
                            # Non-recurrent 모델용 로직
                            loss, pred_scores = self.model(
                                batched_data["input_ids"], batched_data["video_feature"],
                                batched_data["input_mask"], batched_data["token_type_ids"],
                                batched_data["input_labels"]
                            )
                            pred_scores_list = [pred_scores]
                            input_labels_list = [batched_data["input_labels"]]
                            new_memory = None

                    # --- (C) 역전파 (Backpropagation) ---

                    loss = loss / accum_steps
                    grad_norm = None
                    if self.cfg.fp16_train:
                        self.grad_scaler.scale(loss).backward()
                        # Scaling이나 Clipping은 영상 전체가 끝난 후 수행
                    else:
                        loss.backward()

                    if (t + 1) % accum_steps == 0 or (t + 1) == max_steps:
                        if self.cfg.fp16_train:
                            if self.cfg.train.clip_gradient != -1:
                                self.grad_scaler.unscale_(self.optimizer)
                                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.clip_gradient)
                            self.grad_scaler.step(self.optimizer)
                            self.grad_scaler.update()
                        else:
                            if self.cfg.train.clip_gradient != -1:
                                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.clip_gradient)
                            self.optimizer.step()
                        self.optimizer.zero_grad()
                    
                    # --- (D) 다음 스텝을 위한 메모리 업데이트 ---
                    if self.cfg.recurrent and new_memory is not None:
                        current_memory = [m.detach() for m in new_memory]
                    
                    # --- (E) 통계 집계 ---
                    total_loss += loss.item() * accum_steps
                    n_correct = 0
                    n_word = 0

                    for pred, gold in zip(pred_scores_list, input_labels_list):
                        n_correct += cal_performance(pred, gold)
                        valid_label_mask = gold.ne(RecursiveCaptionDataset.IGNORE)
                        n_word += valid_label_mask.sum().item()
                    n_word_total += n_word
                    n_word_correct += n_correct

                # if self.cfg.fp16_train:
                #     if self.cfg.train.clip_gradient != -1:
                #         self.grad_scaler.unscale_(self.optimizer)
                #         grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.clip_gradient)
                #     self.grad_scaler.step(self.optimizer)
                #     self.grad_scaler.update()
                # else:
                #     if self.cfg.train.clip_gradient != -1:
                #         grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.clip_gradient)
                #     self.optimizer.step()
                # # -----------------------------------------------------------
                # # [영상 전체 처리 완료] Optimizer Step (가중치 업데이트)
                # # -----------------------------------------------------------
                # self.hook_post_forward_step_timer()  # hook for step timing (위치 조정 가능)

                
                
                if self.ema is not None:
                    self.ema(self.model, self.state.total_step)

                # (Gradient logging 등은 루프 마지막에 수행)
                if grad_norm is not None:
                    self.metrics.update_meter(MMeters.GRAD, grad_norm)

                additional_log = f" Grad {self.metrics.meters[MMeters.GRAD].avg:.2f}"
                self.hook_post_backward_step_timer()

                current_lr = self.optimizer.get_lr()[0]
                self.hook_post_step(step, loss, current_lr, additional_log=additional_log,
                                    disable_grad_clip=True)

            # log train statistics
            loss_per_word = 1.0 * total_loss / max(n_word_total, 1)
            accuracy = 1.0 * n_word_correct / max(n_word_total, 1)
            self.metrics.update_meter(MMeters.TRAIN_LOSS_PER_WORD, loss_per_word)
            self.metrics.update_meter(MMeters.TRAIN_ACC, accuracy)
            # return loss_per_word, accuracy

            # ---------- validation ----------
            do_val = self.check_is_val_epoch()

            is_best = False
            if do_val:
                # run validation including with ground truth tokens and translation without any text
                _val_loss, _val_score, is_best, _metrics = self.validate_epoch(val_loader)

            # save the EMA weights
            ema_file = self.exp.get_models_file_ema(self.state.current_epoch)
            th.save(self.ema.state_dict(), str(ema_file))

            # post-epoch hook: scheduler, save checkpoint, time bookkeeping, feed tensorboard
            self.hook_post_train_and_val_epoch(do_val, is_best)

        # show end of training log message
        self.hook_post_train()
        
    @th.no_grad()
    def validate_epoch(self, data_loader: data.DataLoader) -> (
            Tuple[float, float, bool, Dict[str, float]]):
        """
        Run both validation and translation.
        [최종 수정] Validation 속도 최적화 + 진행 상황 로그 추가
        """
        self.hook_pre_val_epoch()  # pre val epoch hook
        forward_time_total = 0
        total_loss = 0
        n_word_total = 0
        n_word_correct = 0

        if self.ema is not None:
            self.ema.assign(self.model)

        batch_res = {"version": "VERSION 1.0", "results": defaultdict(list),
                     "external_data": {"used": "true", "details": "ay"}}
        dataset: RecursiveCaptionDataset = data_loader.dataset

        # [설정] 검증할 최대 클립 수 (속도 향상을 위해 20개로 제한)
        EVAL_MAX_CLIPS = 20 

        num_steps = 0
        pbar = tqdm(total=len(data_loader), desc=f"Validate epoch {self.state.current_epoch}")
        
        for _step, batch in enumerate(data_loader):
            self.hook_pre_step_timer()

            # [수정] 배치 데이터 파싱
            batch_clips_lists = batch[0] # List[List[Dict]] (영상별 클립 리스트)
            batch_meta_lists = batch[2]  # List[List[Dict]] (영상별 메타 리스트)
            batch_size = len(batch_clips_lists)

            # ====================================================
            # Part 1: Validation Loss 계산 (Parallel Batching)
            # ====================================================
            actual_max_steps = max([len(clips) for clips in batch_clips_lists])
            loop_steps = min(actual_max_steps, EVAL_MAX_CLIPS)

            current_memory = None
            
            print(f"\n[Val Loss] Batch {_step} | Calculating Loss for {loop_steps} steps...")

            with autocast(enabled=self.cfg.fp16_val):
                if self.cfg.recurrent:
                    for t in range(loop_steps):
                        
                        # [로그 추가] 진행 상황 출력
                        if t % 10 == 0:
                             print(f"\r  > Loss Step {t}/{loop_steps} ...", end="")

                        # --- Step 배치 구성 ---
                        current_step_inputs = []
                        current_step_gt_sentences = []

                        for b_i in range(batch_size):
                            video_clips = batch_clips_lists[b_i]
                            if t < len(video_clips):
                                current_step_inputs.append(video_clips[t])
                                current_step_gt_sentences.append(video_meta[t]["gt_sentence"])
                            else:
                                # Padding
                                dummy = copy.deepcopy(video_clips[-1])
                                dummy['input_labels'][:] = RecursiveCaptionDataset.IGNORE
                                dummy['input_mask'][:] = 0
                                dummy['input_mask'][0] = 1
                                current_step_inputs.append(dummy)
                                current_step_gt_sentences.append(None)
                        
                        collated_input = default_collate(current_step_inputs)
                        batched_data = prepare_batch_inputs(collated_input, use_cuda=self.cfg.use_cuda,
                                                            non_blocking=self.cfg.cuda_non_blocking)

                        # --- 모델 실행 ---
                        input_ids_list = [batched_data["input_ids"]]
                        video_features_list = [batched_data["video_feature"]]
                        input_masks_list = [batched_data["input_mask"]]
                        token_type_ids_list = [batched_data["token_type_ids"]]
                        input_labels_list = [batched_data["input_labels"]]
                        importance_labels_list = self.build_importance_labels(
                            input_labels_list, [current_step_gt_sentences]
                        )

                        loss, pred_scores_list, new_memory = self.model(
                            input_ids_list, video_features_list, input_masks_list,
                            token_type_ids_list, input_labels_list,
                            importance_labels_list=importance_labels_list,
                            past_memory=current_memory
                        )
                        
                        # --- 메모리 업데이트 ---
                        if new_memory is not None:
                            current_memory = new_memory 

                        # --- 통계 ---
                        total_loss += loss.item()
                        
                        n_correct = 0
                        for pred, gold in zip(pred_scores_list, input_labels_list):
                            n_correct += cal_performance(pred, gold)
                            valid_label_mask = gold.ne(RecursiveCaptionDataset.IGNORE)
                            n_word_total += valid_label_mask.sum().item()
                        n_word_correct += n_correct
            
            print(f" -> Loss Calc Done.")

            # ====================================================
            # Part 2: Translation (Generation)
            # ====================================================
            print(f"[Val Gen] Batch {_step} | Generating Captions for {batch_size} videos...")

            for b_i in range(batch_size):
                
                # [로그 추가] 영상 단위 진행 상황 출력
                print(f"\r  > Generating Video {b_i + 1}/{batch_size} (Max {EVAL_MAX_CLIPS} clips)...", end="")

                video_clips = batch_clips_lists[b_i][:EVAL_MAX_CLIPS] 
                video_meta = batch_meta_lists[b_i][:EVAL_MAX_CLIPS]
                
                if not video_clips: continue

                vid_input_ids = []
                vid_features = []
                vid_masks = []
                vid_token_types = []
                
                for clip in video_clips:
                    c_data = prepare_batch_inputs(clip, use_cuda=self.cfg.use_cuda)
                    vid_input_ids.append(c_data["input_ids"].unsqueeze(0))
                    vid_features.append(c_data["video_feature"].unsqueeze(0))
                    vid_masks.append(c_data["input_mask"].unsqueeze(0))
                    vid_token_types.append(c_data["token_type_ids"].unsqueeze(0))
                
                model_inputs = [
                    vid_input_ids, vid_features, vid_masks, vid_token_types
                ]
                
                # 번역 실행
                dec_seq_list = self.translator.translate_batch(
                    model_inputs, use_beam=self.cfg.use_beam, recurrent=True,
                    untied=False, xl=self.cfg.xl
                )
                
                for step_idx, step_batch in enumerate(dec_seq_list):
                    generated_ids = step_batch[0].cpu().tolist()
                    cur_meta = video_meta[step_idx]
                    sent = dataset.convert_ids_to_sentence(generated_ids)
                    
                    batch_res["results"][cur_meta["name"]].append({
                        "sentence": sent,
                        "timestamp": cur_meta["timestamp"],
                        "gt_sentence": cur_meta["sentence"]
                    })

            print(" -> Generation Done.")

            # End of Step
            self.hook_post_forward_step_timer()
            forward_time_total += self.timedelta_step_forward
            num_steps += 1

            if self.cfg.debug:
                break
            pbar.update()
        
        pbar.close()

        # ---------- Validation Done ----------
        batch_res["results"] = self.translator.sort_res(batch_res["results"])

        eval_mode = self.cfg.dataset_val.split
        file_translation_raw = self.exp.get_translation_files(self.state.current_epoch, eval_mode)
        json.dump(batch_res, file_translation_raw.open("wt", encoding="utf8"))

        reference_files_map = get_reference_files(self.cfg.dataset_val.name, self.exp.annotations_dir)
        reference_files = reference_files_map[eval_mode]
        reference_file_single = reference_files[0]

        res_lang = evaluate_language_files(file_translation_raw, reference_files, verbose=False, all_scorer=True)
        res_stats = evaluate_stats_files(file_translation_raw, reference_file_single, verbose=False)
        res_rep = evaluate_repetition_files(file_translation_raw, reference_file_single, verbose=False)

        all_metrics = {**res_lang, **res_stats, **res_rep}

        flat_metrics = {}
        for key, val in all_metrics.items():
            if isinstance(val, Mapping):
                for subkey, subval in val.items():
                    flat_metrics[f"{key}_{subkey}"] = subval
                continue
            flat_metrics[key] = val
        for key, val in flat_metrics.items():
            if isinstance(val, (np.float16, np.float32, np.float64)):
                flat_metrics[key] = float(val)

        for result_key, meter_name in TRANSLATION_METRICS.items():
            self.metrics.update_meter(meter_name, flat_metrics[result_key])

        self.logger.info(f"Done with translation, epoch {self.state.current_epoch} split {eval_mode}")
        self.logger.info(", ".join([f"{name} {flat_metrics[name]:.2%}" for name in TRANSLATION_METRICS_LOG]))

        loss_per_word = 1.0 * total_loss / max(n_word_total, 1)
        accuracy = 1.0 * n_word_correct / max(n_word_total, 1)
        self.metrics.update_meter(MMeters.VAL_LOSS_PER_WORD, loss_per_word)
        self.metrics.update_meter(MMeters.VAL_ACC, accuracy)
        
        forward_time_total /= num_steps
        self.logger.info(
            f"Loss {loss_per_word:.5f} Acc {accuracy:.3%} total {timer() - self.timer_val_epoch:.3f}s, "
            f"forward {forward_time_total:.3f}s")

        if self.cfg.val.det_best_field == "cider":
            val_score = flat_metrics["CIDEr"]
        else:
            raise NotImplementedError(f"best field {self.cfg.val.det_best_field} not known")

        is_best = self.check_is_new_best(val_score)
        self.hook_post_val_epoch(loss_per_word, is_best)

        if self.is_test:
            self.metrics.feed_metrics(False, self.state.total_step, self.state.current_epoch)
            metrics_file = self.exp.path_base / f"val_ep_{self.state.current_epoch}.json"
            self.metrics.save_epoch_to_file(metrics_file)
            self.logger.info(f"Saved validation results to {metrics_file}")

        return total_loss, val_score, is_best, flat_metrics
    
    def get_opt_state(self) -> Dict[str, Dict[str, nn.Parameter]]:
        """
        Return the current optimizer and scheduler state.
        Note that the BertAdam optimizer used already includes scheduling.

        Returns:
            Dictionary of optimizer and scheduler state dict.
        """
        return {
            "optimizer": self.optimizer.state_dict()
            # "lr_scheduler": self.lr_scheduler.state_dict()
        }

    def set_opt_state(self, opt_state: Dict[str, Dict[str, nn.Parameter]]) -> None:
        """
        Set the current optimizer and scheduler state from the given state.

        Args:
            opt_state: Dictionary of optimizer and scheduler state dict.
        """
        self.optimizer.load_state_dict(opt_state["optimizer"])
        # self.lr_scheduler.load_state_dict(opt_state["lr_scheduler"])

    def get_files_for_cleanup(self, epoch: int) -> List[Path]:
        """
        Implement this in the child trainer.

        Returns:
            List of files to cleanup.
        """
        return [
            # self.exp.get_translation_files(epoch, split="train"),
            self.exp.get_translation_files(epoch, split="val"),
            self.exp.get_models_file_ema(epoch)]
