"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

# download the tiny shakespeare dataset
# 入力ファイルのパスを設定
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
# 入力ファイルが存在しない場合、tiny shakespeareデータセットをダウンロード
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

# ダウンロードしたデータセットの読み込み
with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
# テキスト中に存在するすべてのユニークな文字を取得
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
# 文字から整数へのマッピングを作成
stoi = { ch:i for i,ch in enumerate(chars) }
# 整数から文字へのマッピングを作成
itos = { i:ch for i,ch in enumerate(chars) }
# エンコーダの定義。文字列を入力として受け取り、整数のリストを出力
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
# デコーダの定義。整数のリストを入力として受け取り、文字列を出力
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
# 訓練データ90%、検証データ10％に分割
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
# 訓練データを整数にエンコード
train_ids = encode(train_data)
# 検証データを整数にエンコード
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
# エンコードされた訓練データをtrain.binファイルに保存
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
# エンコードされた検証データをval.binファイルに保存
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
# メタ情報（語彙サイズ、エンコーダ、デコーダ）を辞書に保存
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
# メタ情報をmeta.pklという名前のファイルに保存
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
