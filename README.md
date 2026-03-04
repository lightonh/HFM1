# HFM1

flowchart TD
  A[rgb: B×3×H×W] --> B1[CFG mask / force_uncond<br/>rgb_cond]
  X[x_t: B×3×H×W] --> X1[self_cond? concat sc<br/>x_cat: B×(C_x+sc_ch)×H×W]

  X1 --> P1[Patchify(x_cat)<br/>→ x_tok_raw: B×N×(C_x+sc_ch)p²]
  P1 --> LX[x_patch_in Linear<br/>→ B×N×D]
  LX --> PX[+ pos_x + stream_emb[x]<br/>→ x_tok: B×N×D]

  B1 --> PR[Patchify(rgb_cond)<br/>→ rgb_tok_raw: B×N×3p²]
  PR --> LR[rgb_patch_in Linear<br/>→ B×N×D]
  LR --> PR2[+ pos_rgb + stream_emb[rgb]<br/>→ rgb_tok0: B×N×D]

  PR2 --> SA[RGB encoder: 6× TransformerEncoderLayer<br/>(norm_first=True)]
  SA --> SP[rgb_spatial: SpatialConvBlock<br/>(tokens↔spatial dwconv7×7 + MLP-res)]
  SP --> RGBTOK[rgb_tok: B×N×D]

  RGBTOK --> G[rgb_global = mean over N<br/>→ B×D]

  T[t: B] --> TE[sinusoidal_timestep_embedding<br/>→ B×D]
  TE --> TM[time_mlp: Linear→GELU→Linear<br/>→ t_emb: B×D]
  K[task_id: B] --> KE[task_emb Embedding(2,D)<br/>→ k_emb: B×D]
  TM --> C1[concat(t_emb,k_emb,rgb_global)<br/>→ B×3D]
  KE --> C1
  G --> C1
  C1 --> CF[cond_fuse: Linear→GELU→Linear<br/>→ c: B×D]

  PX --> BLK[Main blocks: depth× TransformerBlock<br/>(SelfAttn + CrossAttn(x←rgb) + MLP)<br/>+ (every 4 blocks) SpatialConvBlock]
  RGBTOK --> BLK
  CF --> BLK

  BLK --> NO[norm_out LayerNorm(affine=False)]
  CF --> FM[final_mod: SiLU→Linear(2D)<br/>→ shift,scale]
  NO --> MOD[h*(1+scale)+shift]
  MOD --> OP[out_proj Linear<br/>→ eps_tok: B×N×(out_ch p²)]
  OP --> UP[Unpatchify<br/>→ eps: B×out_ch×H×W]
