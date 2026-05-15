# 補助101ノートブック

これらのノートブックは、W&B Experiment Tracking と Artifacts のスタンドアロン入門資料です。メインワークショップの設計図として機能します。W&B が初めての方は、ワークショップの前にこれらを順に進めることを強くお勧めします。

## セットアップ

1. **依存関係のインストール:**
   ```bash
   pip install -r requirements.txt
   ```

2. **環境の設定:**
   `.env` をコピーして、ご自身の値を入力してください:
   ```
   YOUR_NAME=<あなたの名前>
   WANDB_ENTITY=<あなたのチーム名>
   WANDB_PROJECT=SIE-Workshop-Supplementary-Material
   WANDB_BASE_URL=<あなたの W&B インスタンスの URL>
   WANDB_API_KEY=<あなたの API キー>
   ```

3. **ノートブックを順番に実行してください。**

## ノートブック

| ノートブック | 内容 |
|----------|--------|
| W&B_101_Intro_to_Experiment_Tracking | Run、config、history、summary、スカラー/メディアのロギング、sweep、アラート |
| W&B_101_Intro_to_W&B_Artifacts | Artifact の作成、バージョン管理、エイリアス、TTL、リネージ、Reference Artifact、Registry の概要 |

## クリーンアップ

各ノートブックの末尾には、生成されたファイル（`notebook_generated_material/`）をすべて削除するクリーンアップセルがあります。終了したらコメントを解除して実行してください。
