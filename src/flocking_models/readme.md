# モデルの追加方法

モデルはモジュール化されているため、kinetics、social interaction、neighbor algorithm を別個に追加できます。シミュレーションは開放領域（open field）で行い、壁との相互作用はありません。`SimRunner.jl` 内で `Physics/` 以下の各ディレクトリにある `.jl` コードが自動的に `include` されるので、ユーザーは `.jl` を適切なディレクトリに置くだけで Julia 側の登録は完了します。近傍は Voronoi 隣接に固定されており、TOML では選びません。Social が読む `neighbor_buffers.weights` のため、近傍バッファ自体は残しています。

なお、Python 側のパラメータ設定（`src/types.py`、`src/constants/master_constants.py`、
推定器、`src/constants/estimation.py`の最適化境界）は別途追加する必要があります。

---

## 共通仕様

### 配置場所

| カテゴリ       | ディレクトリ |
|---------------|--------------|
| Kinetics      | `jl_src/Physics/Kinetics/` |
| Social        | `jl_src/Physics/Social/` |
| Neighbor      | `jl_src/Physics/Neighbor/` |

`SimRunner.jl` は各ディレクトリ内の `.jl` をファイル名の辞書順で読み込みます。
重要：Physics モジュール間の `import` 依存は避け、各ファイルを自己完結させてください。

### モジュール名・構造体名・TOML の `type`

`ConfigParser.build_component` は **モジュール名とパラメータ構造体名が一致** することを前提に型を解決します。TOML の `type` にはその名前を文字列で書きます。

| カテゴリ | ファイル例 | モジュール名 / 構造体名 | TOML の `type` |
|----------|------------|-------------------------|----------------|
| Kinetics | `Overdamped.jl` | `Dynamics_Overdamped` | `"Dynamics_Overdamped"` |
| Social | `KR2021.jl` | `Soc_KR2021` | `"Soc_KR2021"` |
| Neighbor | `Voronoi.jl` | `Neighbor_Voronoi` | 固定（TOML に書かない） |

命名規則のまとめ:

- **Kinetics**: モジュール名・構造体名に `Dynamics_` プレフィックスを付ける
- **Social**: モジュール名・構造体名に `Soc_` プレフィックスを付ける
- **Neighbor**: モジュール名・構造体名に `Neighbor_` プレフィックスを付ける。実行時は `Neighbor_Voronoi` のみを使う

### パラメータ構造体と TOML

パラメータ構造体の **フィールド名** が TOML のキーになります（`ConfigParser` が `fieldnames` から読み取る）。

```toml
[Social]
type = "Soc_KR2021"
k_alg = 0.4
k_pos = 0.8
m = 0.8
d_eq = 0.1
```

例外: Kinetics の `v_0` は `[Simulation]` の `v_0` から注入されるため、Kinetics セクションには書きません（構造体には `v_0` フィールドが必要）。

### シミュレーションループでの役割分担

1. `update_neighbors!` — 近傍バッファ更新
2. `reset_buffers!` — social バッファをリセット
3. `calculate_social_interactions!` — `social_factor` を書き込み
4. `update_state_by_kinetics!` — 状態（位置・速度・向きなど）を更新

Social は **kinetics が読めるスカラー因子** `social_factor` をバッファに書き、Kinetics がそれを使って状態を更新します。

---

## Kinetics

**参照実装**: `Physics/Kinetics/Overdamped.jl`, `Inertial.jl`

### 定義するもの

1. **パラメータ構造体** `<名前>{T} <: AbstractKinetics`
   - 必須フィールド: `v_0::T`（TOML では `[Simulation]` から供給）
   - その他のパラメータは自由（例: Overdamped の `sigma_theta`、Inertial の `sigma_omega`, `tau`）

2. **状態構造体** `State_<名前>{T} <: AbstractState`
   - 必須フィールド: `positions`, `velocities`, `orientations`（エージェント数分の `Vector`）
   - 角速度モデルの場合は `ang_vel::Vector{T}` を追加（`Simulation.jl` が `hasproperty(states, :ang_vel)` で HDF5 保存を分岐）

3. **必須関数**

   | 関数 | 役割 |
   |------|------|
   | `allocate_states(kinetics, num_agents)` | 状態の初期配列を返す |
   | `update_state_by_kinetics!(kinetics, states, social_buffers, dt, rng)` | 1 ステップ更新 |
   | `copy_state!(dest, src)` | 状態のコピー |

### 注意

- `social_buffers.social_factor[i]` を読み取り、モデルに応じて向き・角速度などに反映する
- バッファ型は `AbstractBuffer` として受け取り、`.social_factor` フィールド名に依存する（インターフェース上の暗黙の契約）
- `import ..Types: AbstractKinetics, AbstractState, AbstractBuffer, allocate_states, update_state_by_kinetics!, copy_state!`

---

## Social interaction

**参照実装**: `Physics/Social/KR2021.jl`, `Null.jl`

### 定義するもの

1. **パラメータ構造体** `<名前> <: AbstractSocial`（パラメータなしも可）

2. **バッファ構造体** `Buffer_<名前>{T} <: AbstractBuffer`
   - **必須フィールド**: `social_factor::Vector{T}`（長さ = エージェント数）
   - kinetics がステップごとに参照する

3. **必須関数**

   | 関数 | 役割 |
   |------|------|
   | `allocate_social_buffers(social, num_agents, T)` | バッファ生成 |
   | `reset_buffers!(buffers)` | 毎ステップ開始時に `social_factor` をゼロクリア |
   | `calculate_social_interactions!(kinetics, social, states, social_buffers, neighbor_buffers, safe_eps)` | 相互作用を計算し `social_factor` に書き込む |

### 注意

- 近傍情報は `neighbor_buffers.weights[i, j]` を使う（`> 0` なら近傍）
- 既存モデルでは、2 次元力を進行方向法線へ射影し、`kinetics.v_0` で割って無次元化している:
  `social_buffers.social_factor[i] = dot(force, SVector(-s, c)) / kinetics.v_0`
- 相互作用なしモデルでも、空のパラメータ構造体と `social_factor = 0` を書くバッファは必要

---

## Neighbor algorithm

近傍は Voronoi 隣接に固定です。`SimRunner` が `Neighbor_Voronoi` を生成し、`run_simulation` に渡します。隣接グラフの計算には `DelaunayTriangulation` を使います（ドロネー辺が Voronoi 隣接に一致します）。

**参照実装**: `Physics/Neighbor/Voronoi.jl`

### 定義するもの

1. **パラメータ構造体** `<名前> <: AbstractNeighborMethod`（パラメータなしも可）

2. **バッファ構造体** `NeighborBuffer_<名前>{T} <: AbstractBuffer`
   - 典型例: `weights::Matrix{T}`（`num_agents × num_agents`）
   - `weights[i, j] > 0` のとき、エージェント `j` が `i` の近傍

3. **必須関数**

   | 関数 | 役割 |
   |------|------|
   | `allocate_neighbor_buffers(neighbor_method, num_agents, T)` | バッファ生成 |
   | `update_neighbors!(neighbor_method, state, neighbor_buffer, rng)` | 毎ステップ、近傍行列を更新 |

### 注意

- Social モジュールは `neighbor_buffers.weights` の存在を前提とする
- Voronoi 実装では行ごとに正規化（和が 1）しているが、0/1 のみでも可。Social 側の実装と整合させる
- `reset_buffers!` は Neighbor には不要（`update_neighbors!` 内で上書き）

---

## Python 側の追加手順（Julia とは別）

1. `src/types.py` の対応する `Literal[...]` にファイル名を追加（`check_model_registration` 用）
2. `src/constants/master_constants.py` の `PARAMETER_SETTINGS` に、各 setting ごとのパラメータ辞書を追加
3. 必要なら `scripts/01_social_interaction/00_prepare_toml.py` を実行して TOML を再生成

---

## その他の実装上の注意

### ビルド時の DelaunayTriangulation

`DelaunayTriangulation` の precompile が重く、メモリ不足で失敗することがあります。その場合はスワップを確保してから実行してください。

```bash
sudo fallocate -l 8G /swapfile_large
sudo chmod 600 /swapfile_large
sudo mkswap /swapfile_large
sudo swapon /swapfile_large
```
