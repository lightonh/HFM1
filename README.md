# HFM1

## Model Diagram

```mermaid
flowchart TD
  A[rgb: B×3×H×W] --> B1[CFG mask / force_uncond\nrgb_cond]
  X[x_t: B×3×H×W] --> X1[self_cond? concat sc\nx_cat: B×(C_x+sc_ch)×H×W]
  <br/>
  X1 --> P1[Patchify(x_cat)\n→ x_tok_raw: B×N×(C_x+sc_ch)p²]
  P1 --> LX[x_patch_in Linear\n→ B×N×D]
  LX --> PX[+ pos_x + stream_emb[x]\n→ x_tok: B×N×D]
  <br/>
  B1 --> PR[Patchify(rgb_cond)\n→ rgb_tok_raw: B×N×3p²]
  PR --> LR[rgb_patch_in Linear\n→ B×N×D]
  LR --> PR2[+ pos_rgb + stream_emb[rgb]\n→ rgb_tok0: B×N×D]
  <br/>
  PR2 --> SA[RGB encoder: 6× TransformerEncoderLayer\n(norm_first=True)]
  SA --> SP[rgb_spatial: SpatialConvBlock\n(tokens↔spatial dwconv7×7 + MLP-res)]
  SP --> RGBTOK[rgb_tok: B×N×D]
  <br/>
  RGBTOK --> G[rgb_global = mean over N\n→ B×D]
  <br/>
  T[t: B] --> TE[sinusoidal_timestep_embedding\n→ B×D]
  TE --> TM[time_mlp: Linear→GELU→Linear\n→ t_emb: B×D]
  K[task_id: B] --> KE[task_emb Embedding(2,D)\n→ k_emb: B×D]
  TM --> C1[concat(t_emb,k_emb,rgb_global)\n→ B×3D]
  KE --> C1
  G --> C1
  C1 --> CF[cond_fuse: Linear→GELU→Linear\n→ c: B×D]
  <br/>
  PX --> BLK[Main blocks: depth× TransformerBlock\n(SelfAttn + CrossAttn(x←rgb) + MLP)\n+ (every 4 blocks) SpatialConvBlock]
  RGBTOK --> BLK
  CF --> BLK
  <br/>
  BLK --> NO[norm_out LayerNorm(affine=False)]
  CF --> FM[final_mod: SiLU→Linear(2D)\n→ shift,scale]
  NO --> MOD[h*(1+scale)+shift]
  MOD --> OP[out_proj Linear\n→ eps_tok: B×N×(out_ch p²)]
  OP --> UP[Unpatchify\n→ eps: B×out_ch×H×W]
  <br/>
