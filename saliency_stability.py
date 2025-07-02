#!/usr/bin/env python
# saliency_stability.py
"""
Crea un ranking de saliencia por fold y calcula la estabilidad entre folds
(Spearman ρ). Guarda los rankings por fold y un heatmap de concordancia.

Versión: 1.1 (Corregida)
"""

import argparse
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from captum.attr import IntegratedGradients
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Tuple, Union, List

# --- PASO 1: Incluir las definiciones de modelos y helpers ---
# Para que el script sea autocontenido, copiamos aquí las clases y funciones necesarias.

# --- Hiper-parámetros usados cuando se entrenó el VAE ---
# Asegúrate de que coincidan con los que puso serentipia8.py
VAE_PARAMS = dict(
    input_channels      = 3,        # o len(args.channels) si variaba
    latent_dim          = 512,
    image_size          = 131,
    final_activation    = "tanh",
    intermediate_fc_dim_config = "quarter",
    dropout_rate        = 0.2,
    use_layernorm_fc    = False,
    num_conv_layers_encoder = 4,
    decoder_type        = "convtranspose",
    num_groups          = 8,        # GroupNorm
)


class ConvolutionalVAE(nn.Module):
    """ Definición exacta de tu VAE, con GroupNorm, para cargar los pesos correctamente. """
    def __init__(
        self, input_channels: int = 6, latent_dim: int = 128, image_size: int = 131,
        final_activation: str = "tanh", intermediate_fc_dim_config: Union[int, str] = "0",
        dropout_rate: float = 0.2, use_layernorm_fc: bool = False,
        num_conv_layers_encoder: int = 4, decoder_type: str = "convtranspose", num_groups: int = 8
    ) -> None:
        super().__init__()
        if num_conv_layers_encoder not in {3, 4}: raise ValueError("num_conv_layers_encoder must be 3 or 4.")
        if decoder_type not in {"upsample_conv", "convtranspose"}: raise ValueError("decoder_type must be 'upsample_conv' or 'convtranspose'.")
        self.latent_dim = latent_dim; self.num_groups = num_groups; encoder_layers: List[nn.Module] = []; curr_ch = input_channels
        base_conv_ch = [max(16, input_channels * 2), max(32, input_channels * 4), max(64, input_channels * 8), max(128, input_channels * 16)]
        conv_ch_enc = [min(c, 256) for c in base_conv_ch][:num_conv_layers_encoder]
        kernels = [7, 5, 5, 3][:num_conv_layers_encoder]; paddings = [1, 1, 1, 1][:num_conv_layers_encoder]; strides = [2, 2, 2, 2][:num_conv_layers_encoder]
        spatial_dims = [image_size]; dim = image_size
        for k, p, s, ch_out in zip(kernels, paddings, strides, conv_ch_enc):
            encoder_layers += [
                nn.Conv2d(curr_ch, ch_out, kernel_size=k, stride=s, padding=p),
                nn.GELU(), nn.GroupNorm(self.num_groups, ch_out), nn.Dropout2d(p=dropout_rate),
            ]; curr_ch = ch_out; dim = ((dim + 2 * p - k) // s) + 1; spatial_dims.append(dim)
        self.encoder_conv = nn.Sequential(*encoder_layers)
        self.final_conv_ch = curr_ch; self.final_spatial_dim = dim; flat_size = curr_ch * dim * dim
        self.intermediate_fc_dim = self._resolve_intermediate_fc(intermediate_fc_dim_config, flat_size)
        if self.intermediate_fc_dim:
            fc_layers = [nn.Linear(flat_size, self.intermediate_fc_dim)]
            if use_layernorm_fc: fc_layers.append(nn.LayerNorm(self.intermediate_fc_dim))
            fc_layers += [nn.GELU(), nn.BatchNorm1d(self.intermediate_fc_dim), nn.Dropout(p=dropout_rate)]
            self.encoder_fc_intermediate = nn.Sequential(*fc_layers); mu_logvar_in = self.intermediate_fc_dim
        else: self.encoder_fc_intermediate = nn.Identity(); mu_logvar_in = flat_size
        self.fc_mu = nn.Linear(mu_logvar_in, latent_dim); self.fc_logvar = nn.Linear(mu_logvar_in, latent_dim)
        if self.intermediate_fc_dim:
            dec_fc_layers = [nn.Linear(latent_dim, self.intermediate_fc_dim)]
            if use_layernorm_fc: dec_fc_layers.append(nn.LayerNorm(self.intermediate_fc_dim))
            dec_fc_layers += [nn.GELU(), nn.BatchNorm1d(self.intermediate_fc_dim), nn.Dropout(p=dropout_rate)]
            self.decoder_fc_intermediate = nn.Sequential(*dec_fc_layers); dec_fc_out = self.intermediate_fc_dim
        else: self.decoder_fc_intermediate = nn.Identity(); dec_fc_out = latent_dim
        self.decoder_fc_to_conv = nn.Linear(dec_fc_out, flat_size); decoder_layers: List[nn.Module] = []
        curr_ch_dec = self.final_conv_ch; target_conv_t_channels = conv_ch_enc[-2::-1] + [input_channels]
        decoder_kernels = kernels[::-1]; decoder_paddings = paddings[::-1]; decoder_strides = strides[::-1]; output_paddings: List[int] = []
        tmp_dim = self.final_spatial_dim
        for i in range(len(decoder_kernels)):
            k, s, p = decoder_kernels[i], decoder_strides[i], decoder_paddings[i]
            target_dim = spatial_dims[len(decoder_kernels) - 1 - i]; op = target_dim - ((tmp_dim - 1) * s - 2 * p + k)
            output_paddings.append(max(0, min(s - 1, op))); tmp_dim = (tmp_dim - 1) * s - 2 * p + k + op
        for i, ch_out in enumerate(target_conv_t_channels):
            decoder_layers += [nn.ConvTranspose2d(curr_ch_dec, ch_out, kernel_size=decoder_kernels[i], stride=decoder_strides[i], padding=decoder_paddings[i], output_padding=output_paddings[i]), nn.GELU() if i < len(target_conv_t_channels) - 1 else nn.Identity()]
            if i < len(target_conv_t_channels) - 1: decoder_layers += [nn.GroupNorm(self.num_groups, ch_out), nn.Dropout2d(p=dropout_rate)]
            curr_ch_dec = ch_out
        if final_activation == "sigmoid": decoder_layers.append(nn.Sigmoid())
        elif final_activation == "tanh": decoder_layers.append(nn.Tanh())
        self.decoder_conv = nn.Sequential(*decoder_layers)
    def _resolve_intermediate_fc(self, cfg: Union[int, str], flat_size: int) -> int:
        if cfg == "0" or cfg == 0: return 0
        if isinstance(cfg, str):
            cfg = cfg.lower()
            if cfg == "half": return flat_size // 2
            if cfg == "quarter": return flat_size // 4
            try: return int(cfg)
            except ValueError: return 0
        return int(cfg)
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_conv(x); h = h.view(h.size(0), -1); h = self.encoder_fc_intermediate(h)
        return self.fc_mu(h), self.fc_logvar(h)

class FullPipelineWrapper(torch.nn.Module):
    def __init__(self, vae, clf_pipeline, device):
        super().__init__(); self.vae = vae
        scaler = clf_pipeline.named_steps['scaler']
        logreg = clf_pipeline.named_steps['model']
        self.scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
        self.scaler_std = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)
        self.logreg_weights = torch.tensor(logreg.coef_, dtype=torch.float32, device=device)
        self.logreg_bias = torch.tensor(logreg.intercept_, dtype=torch.float32, device=device)
    def forward(self, input_tensor, metadata_tensor):
        mu, _ = self.vae.encode(input_tensor)
        combined_features = torch.cat([mu, metadata_tensor], dim=1)
        scaled_features = (combined_features - self.scaler_mean) / self.scaler_std
        logits = torch.matmul(scaled_features, self.logreg_weights.T) + self.logreg_bias
        return torch.sigmoid(logits)

def apply_normalization_params(data_tensor_subset: np.ndarray, norm_params_per_channel_list: list) -> np.ndarray:
    num_subjects, num_selected_channels, num_rois, _ = data_tensor_subset.shape; normalized_tensor_subset = data_tensor_subset.copy(); off_diag_mask = ~np.eye(num_rois, dtype=bool)
    for c_idx_selected in range(num_selected_channels):
        params = norm_params_per_channel_list[c_idx_selected]; mode = params.get('mode', 'zscore_offdiag')
        current_channel_data = data_tensor_subset[:, c_idx_selected, :, :]; scaled_channel_data_subset = current_channel_data.copy()
        if off_diag_mask.any() and params.get('std', 1.0) > 1e-9:
            scaled_channel_data_subset[:, off_diag_mask] = (current_channel_data[:, off_diag_mask] - params.get('mean', 0.0)) / params['std']
        normalized_tensor_subset[:, c_idx_selected, :, :] = scaled_channel_data_subset
    return normalized_tensor_subset

# --- Función de Carga de Artefactos (CORREGIDA) ---
def load_fold_artifacts(fold_dir: Path, device: torch.device):
    """Carga dinámicamente los artefactos de un fold específico."""
    fold_num = fold_dir.name.split('_')[-1]
    print(f"  Cargando artefactos para el fold número {fold_num}...")
    
    # Construcción dinámica de las rutas
    vae_path = fold_dir / f"vae_model_fold_{fold_num}.pt"
    clf_path = fold_dir / f"classifier_logreg_pipeline_fold_{fold_num}.joblib"
    norm_path = fold_dir / "vae_norm_params.joblib"
    test_idx_path = fold_dir / "test_indices.npy"

    # Carga de los ficheros
    vae_state_dict = torch.load(vae_path, map_location=device)
    clf = joblib.load(clf_path)
    norm = joblib.load(norm_path)
    test_idx = np.load(test_idx_path)
    
    # Re-crear el modelo VAE con los parámetros correctos
    vae = ConvolutionalVAE(**VAE_PARAMS)
    vae.load_state_dict(vae_state_dict)
    vae.eval().to(device)
    
    print(f"  Artefactos para el fold {fold_num} cargados.")
    return vae, clf, norm, test_idx

# --- Función para Calcular Saliencia (simplificada) ---
def get_roi_saliency(full_model, norm_params, subject_tensor, metadata_tensor, device):
    """Calcula la saliencia para un único sujeto."""
    input_tensor = torch.from_numpy(apply_normalization_params(subject_tensor, norm_params)).float().to(device)
    ig = IntegratedGradients(full_model)
    attributions, _ = ig.attribute(
        input_tensor,
        additional_forward_args=(metadata_tensor,),
        target=0, return_convergence_delta=True
    )
    saliency_map = attributions.cpu().detach().numpy().squeeze()
    return np.abs(saliency_map).sum(axis=(0, 2)) # Shape -> (131,)

# --- Script Principal ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de Estabilidad de Saliencia Inter-Fold")
    parser.add_argument("--root", type=str, default="./resultados_12_inter", help="Directorio raíz con los resultados de los folds.")
    parser.add_argument("--tensor", type=str, required=True, help="Ruta al fichero .npz del tensor de datos global.")
    parser.add_argument("--metadata", type=str, required=True, help="Ruta al fichero .csv de metadatos.")
    parser.add_argument("--roi_order", type=str, required=True, help="Ruta al fichero .joblib con la lista ordenada de nombres de ROIs.")
    parser.add_argument("--channels", nargs="+", type=int, default=[1, 2, 5], help="Índices de los canales a usar.")
    args = parser.parse_args()

    # --- Setup ---
    root_dir = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Cargar datos una sola vez
    print("Cargando datos globales...")
    data_tensor = np.load(args.tensor)['global_tensor_data']
    full_metadata_df = pd.read_csv(args.metadata)
    full_metadata_df['SubjectID'] = full_metadata_df['SubjectID'].astype(str).str.strip()
    subject_ids_tensor = np.load(args.tensor)['subject_ids'].astype(str)
    tensor_df = pd.DataFrame({'SubjectID': subject_ids_tensor, 'tensor_idx': np.arange(len(subject_ids_tensor))})
    full_metadata_df = pd.merge(tensor_df, full_metadata_df, on='SubjectID', how='left')
    roi_names_ordered = joblib.load(args.roi_order)
    print("Datos cargados.")

    fold_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    all_rankings = {}

    for fold_dir in fold_dirs:
        print(f"▶ Procesando {fold_dir.name}")
        try:
            vae, clf, norm, test_idx = load_fold_artifacts(fold_dir, device)
            full_model = FullPipelineWrapper(vae, clf, device).eval()
            
            # Obtener los metadatos solo para los sujetos de este fold de test
            test_subjects_df = full_metadata_df.iloc[test_idx]

            scores = []
            for _, subject_row in tqdm(test_subjects_df.iterrows(), total=len(test_subjects_df), desc=f"Saliencia {fold_dir.name}"):
                idx = subject_row['tensor_idx']
                
                # Preparar tensores para este sujeto
                subject_tensor = data_tensor[idx][args.channels][None, ...]
                age_val = float(subject_row['Age']); sex_val = 1.0 if subject_row['Sex'] == 'F' else 0.0
                metadata_np = np.array([[age_val, sex_val]], dtype=np.float32)
                metadata_tensor = torch.from_numpy(metadata_np).to(device)

                # Calcular saliencia
                scores.append(get_roi_saliency(full_model, norm, subject_tensor, metadata_tensor, device))

            avg_saliency = np.mean(scores, axis=0)
            
            df = pd.DataFrame({"ROI_Name": roi_names_ordered, "Saliency_Score": avg_saliency})
            df = df.sort_values("Saliency_Score", ascending=False)
            df.to_csv(fold_dir / "roi_saliency_ranking.csv", index=False)
            all_rankings[fold_dir.name] = df["ROI_Name"].values

        except Exception as e:
            print(f"  ERROR procesando {fold_dir.name}: {e}")
            continue

    # --- Cálculo de Concordancia entre Folds ---
    if len(all_rankings) > 1:
        print("\nCalculando concordancia de rankings entre folds...")
        fold_names = list(all_rankings.keys())
        n = len(fold_names)
        corr_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                rho, _ = spearmanr(all_rankings[fold_names[i]], all_rankings[fold_names[j]])
                corr_matrix[i, j] = rho

        corr_df = pd.DataFrame(corr_matrix, index=fold_names, columns=fold_names)
        corr_df.to_csv(root_dir / "saliency_spearman_matrix.csv")

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_df, annot=True, vmin=0, vmax=1, cmap="viridis", fmt=".3f")
        plt.title("Concordancia de Ranking de ROIs entre Folds (Spearman's ρ)")
        plt.tight_layout()
        plt.savefig(root_dir / "saliency_stability_heatmap.png")
        print("✅ Matriz de concordancia y heatmap guardados.")
        plt.show()
    else:
        print("Se procesó menos de 2 folds, no se puede calcular la estabilidad.")