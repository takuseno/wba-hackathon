## モジュールの概要
実装する全脳アーキテクチャの概要は以下のとおりである。
- 海馬モジュール
- 新皮質モジュール
- 基底核モジュール
![image](https://user-images.githubusercontent.com/15951497/29004679-82d0babe-7b06-11e7-97d3-00d3216a8e3d.png)

## 海馬の実装
![image](https://user-images.githubusercontent.com/15951497/29004659-3348a3f8-7b06-11e7-8063-9de335d4c4c4.png)

## 海馬モジュールの機能
### エピソード記憶
海馬の出力部分にNeural Episodic Controlで用いられているDifferential Neural Dictionaryを実装する
これによって、少ない学習で行動選択が行える
NECとは違い、提案モデルはA3Cによって学習を行うので
- キー
    - 海馬台の出力
- 値
    - 行動の選択確率、基底核の推定した価値

参考
- https://arxiv.org/abs/1703.01988

### 空間認知
CA1に空間認知に関わる細胞が多い
今回の提案モデルではCA1で補助タスクを用いて
それらの再現と探索タスクの性能の向上を目指す
- 場所細胞
    - 訪れた場所かどうかを推定する教師あり学習 (Toxy)
- グリッド細胞
    - 移動量を推定する教師あり学習
    - オドメトリ、CNNで抽出した情報、教師信号としてオドメトリ(運動指令のコピーもしくはFB)
- ヘッドディレクション細胞 
    - 頭部方向を推定する教師あり学習
    - ニューラルネットorCNN(未定)
    - オドメトリ、CNNで抽出した情報、教師信号として頭部情報 (rotationが返ってくるので学習できる)

参考
- https://arxiv.org/abs/1611.03673
- https://arxiv.org/abs/1611.05397

## 海馬モジュールの構造的特徴
### 歯状回
歯状回では新生ニューロンが誕生することがわかっている。
今回のモデルではProgressive Neural Networkを用いてインクリメンタルに
ネットワークが拡張できるように実装を行う。

<img src="https://user-images.githubusercontent.com/15951497/29004759-36be7560-7b08-11e7-8b64-4ae079f4b57f.png" width=300px>

参考
- https://arxiv.org/abs/1606.04671

### CA3
CA3では再帰的な神経投射が行われている。
今回のモデルではRCモデルの一種であるEcho State Networkを用いることで
時系列データを扱い、連想記憶装置としての機能も実装する。

<img src="https://user-images.githubusercontent.com/15951497/29004760-3b25c4be-7b08-11e7-9425-9516b3c7614a.png" width=500px>


## 学習方法
<img src="https://user-images.githubusercontent.com/15951497/29004672-66945d60-7b06-11e7-9d82-25e5e7cc52f9.png" width=300px>
「海馬モジュール」「基底核モジュール」「新皮質モジュール」はA3Cによって学習を行う。
この際、Actorは「海馬モジュール」、Criticは「基底核モジュール」に該当する。
A3Cを用いることで、従来手法(NEC)では実現できなかったRNNの実装を可能とする。また、基底核がTD誤差の計算を行い、海馬は行動選択をしている観点からもDQNではなくA3Cで学習を行うことは妥当だと考えられる。

参考
- https://arxiv.org/abs/1602.01783

## 資料
- [google Slides](https://docs.google.com/presentation/d/1NJXCbx_ijxVEr8nm_keUcXVV7ig0HjMHfiMvFoWSy58/edit#slide=id.g2426f9f589_0_37)
