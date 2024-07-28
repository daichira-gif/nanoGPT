"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

nanoGPTの解説記事
https://qiita.com/shibu_phys/items/c69665912eb60fd87e4c#43-block-class

"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

#レイヤー正規化を実装したクラス
"""
    LayerNormの目的:
        モデルのトレーニングを安定化させ、学習速度を向上させるために使用されます。
        各層の出力を正規化することで、勾配消失や勾配爆発といった問題を軽減します。
    動作原理:
        各層の出力を、層ごとに平均と標準偏差を計算して正規化します。
        具体的には、入力ベクトルの各要素から平均を引き、標準偏差で割ることで正規化を行います。
"""
class LayerNorm(nn.Module):
    #biasオプションによってバイアスを持つかどうかを指定
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

#シーケンス内の位置情報を考慮しながら自己注意機構を実行するクラス
class CausalSelfAttention(nn.Module):
    #キー、クエリ、値の射影を行い、selfattentionを計算

    def __init__(self, config):
        super().__init__()
        #config.n_embdがconfig.n_headで割り切れることを確認
        assert config.n_embd % config.n_head == 0
        
        # key, query, value projections for all heads, but in a batch
        #（key,query,valueのベクトルをまとめて生成するため）入力の埋め込み次元を3倍にして、キー、クエリ、バリューのベクトルを生成する線形変換を定義
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # output projection
        #アテンションの出力を元の埋め込み次元に戻すための線形変換を定義
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # regularization
        #ドロップアウトを使用して過学習を防ぐための正則化を行います。attn_dropoutはアテンションの重みに適用され、resid_dropoutは残差接続に適用されます。
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        #ヘッドの数、埋め込み次元、ドロップアウト率を保存
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        #PyTorch 2.0以上でサポートされるフラッシュアテンションの使用可否を確認。サポートされていない場合、警告を表示
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            #因果マスクの登録: 未来の情報を参照しないよう、入力シーケンスの左側にのみアテンションが適用されるようにする
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        #入力のサイズを取得: 入力テンソルxのサイズを取得し、バッチサイズB、シーケンス長T、埋め込み次元Cに分解
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        #クエリ、キー、バリューの計算: 線形変換self.c_attnを用いて、入力xからクエリq、キーk、バリューvを計算。これらは埋め込み次元Cで分割される
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        
        #キー、クエリ、バリューをそれぞれのヘッドに分割し、次元を変換。transposeを用いてヘッドの次元をバッチ次元の前に移動
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #フラッシュアテンションの使用: PyTorch 2.0以上でサポートされるフラッシュアテンションを使用して、効率的にアテンションを計算。因果関係を考慮したアテンションを適用
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        # フラッシュアテンションが使用できない場合、手動でアテンションを計算。
        # クエリとキーのドット積を計算し、スケーリングしてからソフトマックスを適用。ドロップアウトを適用し、最終的にバリューと掛け合わせます。
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # ヘッドごとの出力を再び結合し、元の次元に戻す。
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        # 出力プロジェクション: アテンションの出力を線形変換し、ドロップアウトを適用して最終的な出力を生成
        y = self.resid_dropout(self.c_proj(y))
        return y

#多層パーセプトロン（MLP）を実装したクラス
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        #入力の拡張: 入力の埋め込み次元config.n_embdを4倍に拡張する線形変換を定義
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        #GELU（Gaussian Error Linear Unit）活性化関数を定義
        self.gelu    = nn.GELU()
        #拡張された次元を元の埋め込み次元に戻す線形変換を定義
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        #過学習を防ぐためのドロップアウトを定義
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

#トランスフォーマーのブロックを構成するクラス
class Block(nn.Module):
    #レイヤー正規化、selfattention、MLPを順に適用
    
    #初期化関数
    def __init__(self, config):
        super().__init__()
        # LayerNormレイヤーを初期化（第一層）
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        # 自己注意レイヤーを初期化
        self.attn = CausalSelfAttention(config)
        # LayerNormレイヤーを初期化（第二層）
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        # MLPレイヤーを初期化
        self.mlp = MLP(config)

    def forward(self, x):
        # LayerNormと自己注意を適用し、元の入力と加算
        x = x + self.attn(self.ln_1(x))
        # LayerNormとMLPを適用し、再度元の入力と加算
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
#GPTモデルの設定を定義するデータクラス
class GPTConfig:
    """GPTモデルの構成設定を定義するデータクラス。
    このクラスは、GPTモデルの構成設定を管理するためのものです。ブロックサイズ、語彙サイズ、レイヤーの数、アテンションヘッドの数、埋め込み次元の数、ドロップアウト率、バイアスの有無などの設定が含まれています。これらの設定は、モデルのトレーニングや推論時に使用されます

    Attributes:
        block_size (int): ブロックサイズ。デフォルトは1024。
        vocab_size (int): 語彙サイズ。GPT-2の語彙サイズは50257で、効率のため64の倍数にパディングされて50304になっている。
        n_layer (int): レイヤーの数。デフォルトは12。
        n_head (int): アテンションヘッドの数。デフォルトは12。
        n_embd (int): 埋め込み次元の数。デフォルトは768。
        dropout (float): ドロップアウト率。デフォルトは0.0。
        bias (bool): 線形層とLayerNormにバイアスを使用するか。Trueの場合はGPT-2のようにバイアスを使用。Falseの場合は少し良くて高速。
    """
    #ブロックサイズ、語彙サイズ、層数、ヘッド数、埋め込み次元などのパラメータを指定
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

#GPTモデルの設定を保持するデータクラス
class GPT(nn.Module):
    #トークン埋め込み、位置埋め込み、ブロックのスタック、最終的な出力の射影を含む

    def __init__(self, config):
        """モデルの初期化。
        モデルの各部分（埋め込み、トランスフォーマーブロック、最終LayerNorm、言語モデルヘッド）を初期化し、重みを適切に設定します。

        Args:
            config (GPTConfig): GPTモデルの設定。
        """
        super().__init__()
        # 設定値の確認
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # トランスフォーマーの各コンポーネントを初期化
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # トークンの埋め込み
            wpe = nn.Embedding(config.block_size, config.n_embd), # 位置の埋め込み
            drop = nn.Dropout(config.dropout), # ドロップアウト層
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # トランスフォーマーブロックのリスト
            ln_f = LayerNorm(config.n_embd, bias=config.bias), # 最終的なLayerNorm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # 言語モデルのヘッド（埋め込み次元から語彙サイズへの線形変換）の定義
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # 重みのタイイング（言語モデルのヘッドとトークン埋め込みの重みを共有）
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        # 重みの初期化
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        # GPT-2論文に基づき、残差プロジェクションの重みを特別に初期化
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        # パラメータ数の報告: モデルのパラメータ数を報告
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    # パラメータ数の取得: モデルのパラメータ数を取得。デフォルトでは、位置埋め込みのパラメータを除外
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    # 重みの初期化メソッド: 線形層と埋め込み層の重みを初期化
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):    
        # 順伝播メソッド: 入力インデックスidxのデバイスとサイズを取得し、シーケンス長がブロックサイズを超えないことを確認、位置エンコーディングを生成
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        # モデルの順伝播: トークン埋め込みと位置埋め込みを取得し、ドロップアウトを適用します。各トランスフォーマーブロックを順に適用し、最終的なLayerNormを適用します。
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # 損失の計算: 目標ターゲットが与えられた場合、クロスエントロピー損失を計算。推論時には、最後の位置のみを使用してロジットを計算
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        # 出力の返却: ロジットと損失を返します。
        return logits, loss
    
    # モデルのブロックサイズを必要に応じて小さく調整するために使用される関数
    # モデルを特定のニーズに合わせてカスタマイズすることが可能になります。
    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        # 新しいブロックサイズが現在のブロックサイズを超えていないことを確認した後、モデルの設定と位置埋め込みの重みを更新します。
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])        
        # 各トランスフォーマーブロックの注意機構のバイアスも新しいブロックサイズに合わせて調整
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """事前訓練済みのモデルをロードして初期化するクラスメソッド。
        指定されたタイプの事前訓練済みGPTモデルをロードし、必要に応じてドロップアウト率をオーバーライドします。
        モデルの各レイヤー数、ヘッド数、埋め込み次元数は、モデルタイプに基づいて設定されます。

        Args:
            model_type (str): モデルのタイプ（'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'のいずれか）。
            override_args (dict, optional): ドロップアウト率などのオーバーライド引数。デフォルトはNone。

        Returns:
            GPT: 初期化されたGPTモデル。
        """
        # モデルタイプの確認
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        # オーバーライド可能な引数がdropoutのみであることを確認
        override_args = override_args or {} # default to empty dict
        assert all(k == 'dropout' for k in override_args)
        # 事前訓練済みモデルのインポート: transformersライブラリからGPT2LMHeadModelをインポートし、モデルタイプを表示
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        # モデル設定の決定: model_typeに基づいて、レイヤー数、ヘッド数、埋め込み次元数を設定します。
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        # 固定設定の適用: 語彙サイズ、ブロックサイズ、バイアスの設定を固定
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        # ドロップアウト率のオーバーライド: override_argsにdropoutが含まれている場合、その値を設定
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        # モデルの初期化: GPTConfigを用いてモデルを初期化し、状態辞書を取得します。attn.biasに関するキーを除外
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        # Huggingfaceモデルの初期化: GPT2LMHeadModelを用いて事前訓練済みモデルをロードし、その状態辞書を取得
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        # パラメータのコピー: パラメータが一致することを確認しながら、Huggingfaceモデルのパラメータをコピーします。Conv1Dモジュールの重みは転置してコピー
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        # 初期化されたモデルを返す
        return model

    # メソッドの定義: 最適化アルゴリズムを設定するメソッド
    # 重み減衰、学習率、ベータ値、デバイスタイプを引数として受け取ります
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        # パラメータの取得: モデルの全てのパラメータを辞書形式で取得
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # 勾配が必要なパラメータのフィルタリング: 勾配が必要なパラメータのみをフィルタリングします。
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # パラメータのグループ化: 2次元以上のパラメータ（重み）は重み減衰を適用し、それ以外のパラメータ（バイアスやLayerNorm）は適用しません。
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        # 最適化グループの作成: 重み減衰を適用するパラメータと適用しないパラメータのグループを作成
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # パラメータ数の表示: 重み減衰を適用するパラメータと適用しないパラメータの数を表示します。
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # AdamWオプティマイザの作成: AdamWオプティマイザを作成し、利用可能な場合は融合バージョンを使用
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        # 作成したオプティマイザを返す
        return optimizer

    # メソッドの定義: モデルのフロップス利用率（MFU）を推定するメソッド
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        # フロップスの計算: 各イテレーションで実行されるフロップスの数を計算
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # MFUの計算と返却: 達成されたフロップスをA100 GPUのピークフロップスと比較してMFUを計算し、返す。
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    # メソッドの定義: 条件付きシーケンスidxを受け取り、max_new_tokens回シーケンスを補完するメソッド。
    # 推論モードで実行することを推奨
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        
        Logitsの概要
        定義:
        logitsは、ニューラルネットワークの最終層の出力であり、ソフトマックス関数に入力される前の値です1。
        これらの値は、各クラスに対するスコアを表しており、まだ確率には変換されていません。
        使用例:
        例えば、画像分類タスクにおいて、モデルが3つのクラス（猫、犬、鳥）を予測する場合、logitsは各クラスに対するスコアを出力します。これらのスコアは、ソフトマックス関数を通すことで確率に変換されます。
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # # シーケンスのトリミング: シーケンスが長すぎる場合、ブロックサイズに合わせてトリミング
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # モデルの順伝播: モデルを順伝播させてロジットを取得
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # ロジットのスケーリング: 最後のステップのロジットを取得し、温度でスケーリング
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # トップkのロジットの選択: 必要に応じて、ロジットをトップkのオプションにトリミング
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            # ソフトマックスの適用: ロジットにソフトマックスを適用して確率に変換
            probs = F.softmax(logits, dim=-1)
            # 次のトークンのサンプリング: 確率分布から次のトークンをサンプリング
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # シーケンスの更新: サンプリングされたトークンをシーケンスに追加し、続行
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            
        # 生成されたシーケンスを返す。
        return idx
