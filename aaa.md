사용자님의 **80GB VRAM** 환경을 십분 활용하여, \*\*Pass 1(과거 참조)\*\*과 \*\*Pass 2(미래 참조)\*\*를 동시에 학습시키는 **End-to-End 완성 코드**입니다.

이 코드는 단순한 복사-붙여넣기가 아니라, \*\*데이터가 흐르는 모든 통로(`Layer` -\> `Encoder` -\> `Transformer`)\*\*를 뚫어놓은 버전입니다.

아래 순서대로 **`mart/model.py`** 파일의 해당 클래스들을 덮어씌우시면 됩니다.

-----

### 1\. `BertLayerWithMemory` 수정

**역할:** 실제로 미래 기억(`future_m`)을 받아서 현재 정보와 합체(Concat)하고 어텐션을 수행하는 부품입니다.

```python
class BertLayerWithMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention = BertAttention(config)
        self.memory_initilizer = MemoryInitializer(config)
        self.memory_updater = MemoryUpdater(config)
        self.memory_augmented_attention = BertSelfAttention(config)
        self.hidden_intermediate = BertIntermediate(config)
        self.memory_projection = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output = BertOutput(config)

    def forward(self, prev_m, hidden_states, attention_mask, future_m=None):
        """
        Args:
            future_m: (N, M, D) - 미래 시점에서 가져온 메모리 (Type 1일 때만 들어옴)
        """
        max_v_len, max_t_len = self.config.max_v_len, self.config.max_t_len
        
        # 1. Self-Attention (현재 정보끼리 확인)
        shifted_self_mask = make_pad_shifted_mask(attention_mask, max_v_len, max_t_len)
        attention_output = self.attention(hidden_states, shifted_self_mask)
        intermediate_output = self.hidden_intermediate(attention_output)

        # 2. 메모리 초기화 (첫 스텝일 경우)
        if prev_m is None:
            init_memory_mask = make_video_only_mask(attention_mask, max_v_len)
            prev_m = self.memory_initilizer(intermediate_output, init_memory_mask)

        # 3. 메모리 업데이트 (다음 스텝을 위해 기억 갱신)
        updated_m = self.memory_updater(prev_m, intermediate_output, attention_mask)

        # ==============================================================================
        # [핵심 수정] Memory Augmented Attention: 과거 + 현재 + [미래] 참조
        # ==============================================================================
        
        # 기본: [과거 메모리, 현재 정보]
        concat_list = [prev_m, intermediate_output]
        
        # [Type 1] 미래 메모리가 있으면 추가!
        if self.config.type == 1 and future_m is not None:
            concat_list.append(future_m)

        # 하나로 합치기 (Key, Value로 사용됨)
        concat_mh = torch.cat(concat_list, dim=1) 

        # 마스크 처리
        bsz = prev_m.shape[0]
        n_memory_cells = prev_m.shape[1]
        
        # 기본 마스크: [과거(1), 현재(attention_mask)]
        raw_mask_list = [attention_mask.new_ones(bsz, n_memory_cells), attention_mask]
        
        # 미래 마스크 추가: [미래(1)] -> 미래 정보도 다 볼 수 있게 1로 설정
        if self.config.type == 1 and future_m is not None:
             raw_mask_list.append(attention_mask.new_ones(bsz, future_m.shape[1]))

        raw_memory_attention_mask = torch.cat(raw_mask_list, -1)

        # Shifted Mask 생성 (MART 기존 로직 호환)
        # 여기서 memory_len은 과거 메모리 길이만 넣어주는 게 안전함 (함수 내부 로직상)
        # 하지만 concat_mh 전체 길이에 맞춰 마스크가 생성되어야 함.
        # 기존 함수(make_pad_shifted_mask)를 쓰되, 전체 길이를 맞추기 위해 꼼수를 씀.
        
        # 간단한 해결책: 메모리(과거+미래) 부분은 다 보여주고(1), 현재 부분만 마스킹 처리
        # concat_mh의 구조: [Past(M) | Curr(L) | Future(M)]
        
        total_len = concat_mh.shape[1]
        curr_len = intermediate_output.shape[1]
        
        # (N, L, Total_Len) 크기의 마스크 생성
        # 1. 일단 다 1로 채움
        final_mask = attention_mask.new_ones(bsz, curr_len, total_len)
        
        # 2. 현재(Curr) 부분에 대해서만 기존 attention_mask 적용
        # Past가 앞에 있으므로, Curr의 시작 인덱스는 n_memory_cells
        # final_mask[:, :, n_memory_cells : n_memory_cells+curr_len] = ... (복잡함)
        
        # --> 기존 코드 활용:
        # raw_memory_attention_mask는 (N, Total_Len) 형태임 (1, 1..1, 0, 0.. 1, 1)
        # 이걸 make_pad_shifted_mask에 넣으면 (N, L, Total_Len)이 나옴.
        # 단, memory_len 인자를 '과거 메모리 길이'만 주면 됨.
        
        memory_attention_mask = make_pad_shifted_mask(
            raw_memory_attention_mask, max_v_len, max_t_len, memory_len=n_memory_cells)
            
        # 4. 어텐션 수행 (Query: 현재, Key/Value: 과거+현재+미래)
        memory_attention_output = self.memory_augmented_attention(
            intermediate_output, concat_mh, concat_mh, memory_attention_mask)
            
        memory_attention_output = self.memory_projection(memory_attention_output)
        layer_output = self.output(memory_attention_output, attention_output)

        return updated_m, layer_output
```

-----

### 2\. `BertEncoderWithMemory` 수정

**역할:** 지휘자(Transformer)가 준 `future_m`을 각 층의 `BertLayer`로 배달하는 역할입니다.

```python
class BertEncoderWithMemory(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BertLayerWithMemory(config) for _ in range(config.num_hidden_layers)])

    def forward(self, prev_ms, hidden_states, attention_mask, future_ms=None, output_all_encoded_layers=True):
        """
        Args:
            future_ms: [(N, M, D)] * num_layers (각 층별 미래 메모리 리스트)
        """
        all_encoder_layers = []
        for layer_idx, layer_module in enumerate(self.layer):
            
            # 해당 레이어에 맞는 미래 메모리 꺼내기 (없으면 None)
            layer_future_m = future_ms[layer_idx] if future_ms is not None else None
            
            # 레이어 실행 (future_m 전달!)
            prev_ms[layer_idx], hidden_states = layer_module(
                prev_ms[layer_idx], hidden_states, attention_mask, future_m=layer_future_m
            )
            
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            
        return prev_ms, all_encoder_layers
```

-----

### 3\. `RecursiveTransformer` 수정 (최종 완성)

**역할:** 1차 주행(과거) -\> 2차 주행(미래 참조) -\> Loss 합산(Joint Training)을 총괄합니다.

```python
class RecursiveTransformer(nn.Module):
    def __init__(self, cfg: MartConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.type == 1: 
            print('>>> [Type 1] Future-Aware Memory Training Activated! (Joint Training)')
            
        self.embeddings = BertEmbeddingsWithVideo(cfg, add_postion_embeddings=True)
        self.encoder = BertEncoderWithMemory(cfg)
        
        decoder_classifier_weight = self.embeddings.word_embeddings.weight \
            if self.cfg.share_wd_cls_weight else None
        self.decoder = BertLMPredictionHead(cfg, decoder_classifier_weight)
        
        if self.cfg.label_smoothing != 0:
            self.loss_func = LabelSmoothingLoss(cfg.label_smoothing, cfg.vocab_size, ignore_index=-1)
        else:
            self.loss_func = nn.CrossEntropyLoss(ignore_index=-1)

        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.cfg.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward_step(self, prev_ms, input_ids, video_features, input_masks,
                     token_type_ids, future_m=None):
        """
        Step 실행: future_m을 받아서 encoder로 넘겨줌
        """
        embeddings = self.embeddings(input_ids, video_features, token_type_ids)

        # Encoder에 future_ms 전달
        prev_ms, encoded_layer_outputs = self.encoder(
            prev_ms, embeddings, input_masks, future_ms=future_m, output_all_encoded_layers=False)
            
        prediction_scores = self.decoder(encoded_layer_outputs[-1])
        return prev_ms, encoded_layer_outputs, prediction_scores

    def forward(self, input_ids_list, video_features_list, input_masks_list,
                token_type_ids_list, input_labels_list, return_memory=False):
        
        # ------------------------------------------------------------------
        # [Phase 1] 1차 주행 (Pass 1): 과거 정보만으로 학습 & 미래 메모리 생성
        # ------------------------------------------------------------------
        prev_ms = [None] * self.cfg.num_hidden_layers
        step_size = len(input_ids_list)
        
        future_memories_list = [] # Pass 2를 위한 컨닝페이퍼 (Gradient 포함됨!)
        pass1_scores_list = []    # Pass 1 채점용
        
        # VRAM 80GB니까 no_grad 없이 그냥 돌립니다. (End-to-End 학습)
        for idx in range(step_size):
            prev_ms, _, prediction_scores = self.forward_step(
                prev_ms, input_ids_list[idx], video_features_list[idx],
                input_masks_list[idx], token_type_ids_list[idx]
            )
            # 다음 루프나 Pass 2에서 쓸 수 있게 저장
            # detach() 안 함 -> Pass 2의 Loss가 Pass 1까지 전파됨
            future_memories_list.append([m for m in prev_ms]) 
            pass1_scores_list.append(prediction_scores)

        # 메모리 분석용 리턴 (학습 아닐 때)
        if return_memory:
            return future_memories_list

        # ------------------------------------------------------------------
        # [Phase 2] 2차 주행 (Pass 2): 미래 기억을 훔쳐보며 다시 학습
        # ------------------------------------------------------------------
        pass2_scores_list = []
        prev_ms = [None] * self.cfg.num_hidden_layers # 메모리 초기화
        
        for idx in range(step_size):
            prev_masks = None if idx == 0 else input_masks_list[idx - 1]
            
            # 미래 기억(Future Memory) 가져오기 로직
            current_future_m = None
            if self.cfg.type == 1:
                # 마지막 스텝이 아니면 다음 스텝(idx+1)의 메모리를 가져옴
                if idx < step_size - 1:
                    current_future_m = future_memories_list[idx+1]
                else:
                    # 마지막 스텝은 미래가 없으니 현재(idx)꺼 사용
                    current_future_m = future_memories_list[idx]

            # 2차 주행: future_m을 주입!
            prev_ms, _, prediction_scores = self.forward_step(
                prev_ms, input_ids_list[idx], video_features_list[idx],
                input_masks_list[idx], token_type_ids_list[idx],
                future_m=current_future_m  # <--- [핵심] 미래 정보
            )
            pass2_scores_list.append(prediction_scores)

        # ------------------------------------------------------------------
        # [Phase 3] Loss 계산 (Pass 1 + Pass 2)
        # ------------------------------------------------------------------
        caption_loss = 0.
        
        for idx in range(step_size):
            targets = input_labels_list[idx].view(-1)
            
            # 1. Pass 1 Loss (과거만 보고 맞춘 점수)
            loss_1 = self.loss_func(
                pass1_scores_list[idx].view(-1, self.cfg.vocab_size), targets
            )
            
            # 2. Pass 2 Loss (미래도 보고 맞춘 점수)
            loss_2 = self.loss_func(
                pass2_scores_list[idx].view(-1, self.cfg.vocab_size), targets
            )
            
            # 최종 Loss 합산 (Joint Training)
            caption_loss += (loss_1 + loss_2)

        # 80GB VRAM Flex!
        # inference_only 모드(validate)일 때는 Pass 2 결과(미래반영)를 리턴
        return caption_loss, pass2_scores_list
```