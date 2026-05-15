# W&B Workshop 2026: 実験からプロダクションまで、完全な MLOps を体験する

**関連リソース:**
- [ワークショップレポート - 日本語版](https://wandb.ai/m-bakir/SIE-Workshop-2026/reports/MLOps-in-Practice-Workshop-Guide-Japanese--VmlldzoxNjg5NTYyNA) -- 埋め込みパネル、インタラクティブなクエリ、セクションごとの解説を含む参考資料

- [W&B プロジェクト](https://wandb.ai/m-bakir/SIE-Workshop-2026) -- すべての Run、Artifact、Registry エントリを集約した共有プロジェクト

---

海洋生物の画像分類器を構築しながら、Weights & Biases を使って最初の実験からプロダクションモデルまで、MLOps のライフサイクル全体を学びます。

## このワークショップで行うこと

AQUA 水中データセット（20 種の海洋生物、8,000 枚以上の画像）を使って、ベースラインモデルの学習、ハイパーパラメータ Sweep の実行、そして最良モデルのプロダクションへの昇格までを行います。すべての過程は W&B でエンドツーエンドに追跡されます。

**取り上げるトピック:**
- 実験追跡（Run、config、メトリクス、アラート、commit=False、define_metric）
- 視覚的ロギング（画像、テーブル、ROC 曲線、クラス別メトリクス）
- Artifacts（バージョン管理、リネージ、TTL、Reference Artifacts）
- Run の再開（ID で再開し、シームレスに学習を継続）
- オフラインモード（接続なしで学習し、後から同期）
- Model Registry（ステージング、プロダクションへの昇格）
- Sweeps（ハイパーパラメータ最適化、CLI からの Sweep、並列 agent）
- Automations（Registry イベントからの CI/CD トリガー）
- プログラマティック API（オプション -- Run のクエリ、フィルタ、メタデータ）
- プログラマティックレポート（オプション -- Reports API、PanelGrid）
- SDK 設定リファレンス（オプション -- ネットワーク、git、分散学習）

## はじめに

### 1. Python 環境のセットアップ

既にプロビジョニング済みの環境（JupyterHub カーネルなど）をお持ちの場合は、ステップ 2 に進んでください。

そうでない場合は、仮想環境を作成して依存関係をインストールしてください:

```bash
cd 2026-Workshop
python -m venv workshop
source workshop/bin/activate
pip install -r requirements.txt
```

> **注意:** Jupyter でノートブックを実行する場合は、同じ環境に `jupyter` と `ipykernel` がインストールされていることを確認した上で、カーネルを登録してください:
> ```bash
> pip install jupyter ipykernel
> python -m ipykernel install --user --name wandb_workshop --display-name "W&B Workshop"
> ```

### 2. `.env` ファイルの設定

`workshop_material/.env` を開いて、W&B の認証情報を入力してください:

```
YOUR_NAME=<あなたの名前>
WANDB_ENTITY=your-team-name
WANDB_PROJECT=SIE-Workshop-2026
WANDB_BASE_URL=https://your-wandb-instance.example.com
WANDB_API_KEY=your-api-key-here
```

ワークショップのすべてのファイル（ノートブック、Sweep スクリプト、共有ワーカー）は、この 1 つのファイルから設定を読み込みます。

API キーは **W&B UI > Profile > Settings > API Keys** で確認できます。

### 3. ローカルデータの確認

データセットと事前学習済みのモデル重みは、`workshop_material/` 配下に既にプリロードされているはずです:

```
workshop_material/
  data/
    train/               # 約 6,500 枚の学習画像（20 クラスのサブフォルダ）
    val/                 # 約 800 枚の検証画像
    test/                # 約 800 枚のテスト画像
  pretrained_weights/
    resnet50_imagenet.pth
    efficientnet_b0_imagenet.pth
```

ノートブックはこれらのローカルディレクトリからデータを読み込みます。また、W&B でリネージを追跡するために `use_artifact()` も呼び出します。これにより、何かをダウンロードすることなく、どのデータセットバージョンが使われたかを学習 Run に正確に記録できます。

`data/` または `pretrained_weights/` ディレクトリが見当たらない場合は、ワークショップのファシリテーターに連絡してください。どうしても必要なときは、フォールバックスクリプトを自分で実行することもできます（インターネット接続が必要です）:

```bash
cd admin_setup_only
pip install datasets torch timm Pillow numpy scikit-learn
python prepare_local_data.py
```

### 4. ワークショップノートブックを開く

`workshop_material/aqua_with_wandb.ipynb` を開いて進めてください。ノートブックは `.env` の認証情報を使って W&B 認証を自動的に行います。

## 補助資料

W&B が初めての方は、`supplementary_101_notebooks/` を参照してください。実験追跡と Artifacts／Registry のスタンドアロンの入門資料があります。

## リポジトリ構成

```
2026-Workshop/
  workshop_material/
    aqua_with_wandb.ipynb       # メインのワークショップノートブック（ここから開始）
    workshop_utils.py           # ML 周りの定型コード（データ読み込み、学習ループ）
    .env                        # あなたの W&B 認証情報（初回のみ入力）
    sweep_train.py              # スタンドアロンの Sweep 学習スクリプト（CLI Sweep 用）
    sweep_config.yaml           # Sweep の探索空間の設定
    shared_worker.py            # Shared モードのデモ（マルチプロセスロギング）
    data/                       # プリロード済みのデータセット分割（train/val/test）
    pretrained_weights/         # プリロード済みのモデル重み
  supplementary_101_notebooks/  # W&B のコンセプトを掘り下げる補助資料（オプション）
  admin_setup_only/             # 管理者専用: データセット + モデルのアップロードスクリプト
  requirements.txt              # Python の依存関係
```
