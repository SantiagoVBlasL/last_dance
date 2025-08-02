#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interpretar_fold_gpt.py
===================

Pipeline UNIFICADO de interpretabilidad post-entrenamiento para tu experimento
VAE + Clasificador (CN vs AD). Integra y reemplaza a:

  * run_shap_analysis15.py  (cálculo SHAP del clasificador sobre latentes + metadatos)
  * interpretar_modelo_final_signed.py  (proyección de pesos latentes → saliencia conectómica)

El flujo es:

Entrenamiento (serentipia13.py)  → artefactos por fold  →  interpretar_fold.py (shap|saliency)

Subcomandos:

  • **shap**: Reconstituye los datos de test del fold, calcula valores SHAP de la clase positiva
              para el clasificador indicado y guarda un *shap_pack_*.joblib canónico.

  • **saliency**: Carga el *shap_pack*, calcula pesos para las features latentes (varias estrategias),
                  propaga esos pesos como gradientes a través del VAE para obtener mapas de saliencia
                  por grupo (AD, CN) y su diferencia (AD−CN), y genera rankings de conexiones.

Uso mínimo:

```bash
python interpretar_fold_gpt.py shap \
  --run_dir ./resultados_22 --fold 1 --clf xgb \
  --global_tensor_path /ruta/GLOBAL_TENSOR.npz \
  --metadata_path /ruta/SubjectsData.csv \
  --channels_to_use 1 2 5 \
  --latent_dim 512 --latent_features_type mu \
  --metadata_features Age Sex Manufacturer \
  --num_conv_layers_encoder 4 --decoder_type convtranspose \
  --dropout_rate_vae 0.2 --use_layernorm_vae_fc \
  --intermediate_fc_dim_vae half --vae_final_activation linear \
  --gn_num_groups 32

python interpretar_fold_gpt.py saliency \
  --run_dir ./resultados_22 --fold 1 --clf xgb \
  --global_tensor_path /ruta/GLOBAL_TENSOR.npz \
  --metadata_path /ruta/SubjectsData.csv \
  --roi_order_path ./resultados_22/roi_order_131.joblib \
  --channels_to_use 1 2 5 --latent_dim 512 \
  --num_conv_layers_encoder 4 --decoder_type convtranspose \
  --dropout_rate_vae 0.2 --use_layernorm_vae_fc \
  --intermediate_fc_dim_vae half --vae_final_activation linear \
  --gn_num_groups 32 --top_k 50 --shap_weight_mode ad_vs_cn_diff
```

-------------------------------------------------------------------------------
NOTAS IMPORTANTES
-------------------------------------------------------------------------------
* El script **no re-entrena nada**. Asume que ya corriste `serentipia13.py` y que los artefactos por fold
  existen (pipelines de clasificadores, VAE, norm params, etc.).
* Todos los paths se resuelven a partir de `--run_dir` y `--fold`.
* Se usa `label_mapping.json` guardado en cada fold para identificar la clase positiva (p.ej. AD).
* Se asume que `shap_background_data_{clf}.joblib` guardado durante entrenamiento **ya está procesado**
  (pasado por el preprocessor del pipeline). Si detecta que no lo está, intentará procesarlo.
* Los mapas de saliencia se fuerzan a ser **simétricos** y con **diagonal en cero**, apropiado para matrices de conectividad.

-------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import joblib
import matplotlib
matplotlib.use("Agg")  # backend no interactivo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch

# ---------------------------------------------------------------------------
# Imports de tu repo local
# ---------------------------------------------------------------------------
try:
    from models.convolutional_vae3 import ConvolutionalVAE
except ImportError as e:  # ayuda si se ejecuta fuera del repo raíz
    raise ImportError("No se pudo importar ConvolutionalVAE desde models.convolutional_vae3.\n"
                      "Asegúrate de ejecutar desde la raíz del proyecto o de que PYTHONPATH esté configurado.") from e

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger('interpret')
logging.getLogger('shap').setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Utilidades básicas
# ---------------------------------------------------------------------------

def clean_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Elimina el prefijo "_orig_mod." que añade torch.compile."""
    return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}


def build_vae(vae_kwargs: Dict[str, Any], state_dict_path: Path, device: torch.device) -> ConvolutionalVAE:
    """Construye e inicializa un VAE con pesos entrenados."""
    vae = ConvolutionalVAE(**vae_kwargs).to(device)
    sd = torch.load(state_dict_path, map_location=device)
    vae.load_state_dict(clean_state_dict(sd))
    vae.eval()
    return vae


def unwrap_model_for_shap(model: Any, clf_type: str) -> Any:
    """Extrae el estimador base de un CalibratedClassifierCV cuando aplica."""
    if hasattr(model, 'calibrated_classifiers_') and clf_type in {'xgb', 'gb', 'rf', 'lgbm'}:
        cc = model.calibrated_classifiers_[0]
        if hasattr(cc, 'estimator') and cc.estimator is not None:
            return cc.estimator
        if hasattr(cc, 'base_estimator') and cc.base_estimator is not None:
            return cc.base_estimator
    return model


def _to_sample_feature(sh_vals: Union[np.ndarray, List[np.ndarray]],
                       positive_idx: int,
                       n_samples: int,
                       n_features: int) -> np.ndarray:
    """Devuelve siempre array 2D (samples, features) para la clase positiva."""
    # TreeExplainer binario → array (n_samples, n_features)
    if isinstance(sh_vals, np.ndarray) and sh_vals.ndim == 2 and sh_vals.shape == (n_samples, n_features):
        return sh_vals
    # Tree/Kernal multiclase → list de arrays
    if isinstance(sh_vals, list):
        return sh_vals[positive_idx]
    # Formato 3D → (samples, features, classes)
    if isinstance(sh_vals, np.ndarray) and sh_vals.ndim == 3:
        return sh_vals[:, :, positive_idx]
    raise ValueError(f"Formato SHAP inesperado: type={type(sh_vals)} shape={getattr(sh_vals,'shape',None)}")


# ---------------------------------------------------------------------------
# Normalización (copiado de serentipia13.py para independencia)
# ---------------------------------------------------------------------------

def apply_normalization_params(data_tensor_subset: np.ndarray,
                               norm_params_per_channel_list: List[Dict[str, float]]) -> np.ndarray:
    """Aplica parámetros de normalización guardados por canal (off-diag)."""
    num_subjects, num_selected_channels, num_rois, _ = data_tensor_subset.shape
    normalized_tensor_subset = data_tensor_subset.copy()
    off_diag_mask = ~np.eye(num_rois, dtype=bool)
    if len(norm_params_per_channel_list) != num_selected_channels:
        raise ValueError("# canales en datos != # canales en parámetros de normalización")
    for c_idx, params in enumerate(norm_params_per_channel_list):
        mode = params.get('mode', 'zscore_offdiag')
        if params.get('no_scale', False):
            continue
        current_channel_data = data_tensor_subset[:, c_idx, :, :]
        scaled_channel_data = current_channel_data.copy()
        if off_diag_mask.any():
            if mode == 'zscore_offdiag':
                std = params.get('std', 1.0)
                mean = params.get('mean', 0.0)
                if std > 1e-9:
                    scaled_channel_data[:, off_diag_mask] = (current_channel_data[:, off_diag_mask] - mean) / std
            elif mode == 'minmax_offdiag':
                mn = params.get('min', 0.0)
                mx = params.get('max', 1.0)
                rng = mx - mn
                if rng > 1e-9:
                    scaled_channel_data[:, off_diag_mask] = (current_channel_data[:, off_diag_mask] - mn) / rng
                else:
                    scaled_channel_data[:, off_diag_mask] = 0.0
        normalized_tensor_subset[:, c_idx, :, :] = scaled_channel_data
    return normalized_tensor_subset


# ---------------------------------------------------------------------------
# Carga de datos / artefactos del fold
# ---------------------------------------------------------------------------

def _load_global_and_merge(global_tensor_path: Path, metadata_path: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """Carga tensor global (.npz) + metadata CSV y los une en un DF con tensor_idx."""
    npz = np.load(global_tensor_path)
    tensor_all = npz['global_tensor_data']
    subj_all = npz['subject_ids'].astype(str)
    meta = pd.read_csv(metadata_path)
    meta['SubjectID'] = meta['SubjectID'].astype(str).str.strip()
    tensor_df = pd.DataFrame({'SubjectID': subj_all, 'tensor_idx': np.arange(len(subj_all))})
    merged = tensor_df.merge(meta, on='SubjectID', how='left')
    return tensor_all, merged


def _subset_cnad(merged_df: pd.DataFrame) -> pd.DataFrame:
    return merged_df[merged_df['ResearchGroup_Mapped'].isin(['CN', 'AD'])].reset_index(drop=True)


def _load_label_info(fold_dir: Path) -> Dict[str, Any]:
    p = fold_dir / 'label_mapping.json'
    if p.exists():
        with open(p) as f:
            return json.load(f)
    # fallback: asumir AD=1 CN=0
    log.warning("label_mapping.json no encontrado; se asume CN=0 / AD=1")
    return {'label_mapping': {'CN': 0, 'AD': 1}, 'positive_label_name': 'AD', 'positive_label_int': 1}


# ---------------------------------------------------------------------------
# SHAP (subcomando "shap")
# ---------------------------------------------------------------------------

def cmd_shap(args: argparse.Namespace) -> None:
    fold_dir = Path(args.run_dir) / f"fold_{args.fold}"
    out_dir = fold_dir / 'interpretability_shap'
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"[SHAP] fold={args.fold} clf={args.clf}")

    # Artefactos entrenamiento -------------------------------------------------
    pipe_path = fold_dir / f"classifier_{args.clf}_pipeline_fold_{args.fold}.joblib"
    if not pipe_path.exists():
        raise FileNotFoundError(f"No se encontró el pipeline del clasificador: {pipe_path}")
    pipe = joblib.load(pipe_path)

# --- Carga de datos de background para SHAP ---
    # Se prioriza cargar el fichero de datos CRUDOS (raw), que es el formato
    # que espera el preprocesador para la transformación.
    bg_path_raw = fold_dir / f"shap_background_raw_{args.clf}.joblib"
    
    if bg_path_raw.exists():
        log.info(f"[SHAP] Cargando datos de background CRUDOS desde: {bg_path_raw.name}")
        background_data = joblib.load(bg_path_raw)
    else:
        # Fallback al formato antiguo (pre-procesado) por retrocompatibilidad.
        # Esto puede fallar si el preprocesador ha cambiado.
        bg_path_processed = fold_dir / f"shap_background_data_{args.clf}.joblib"
        log.warning(f"[SHAP] No se encontró 'shap_background_raw_{args.clf}.joblib'. "
                    f"Intentando con el fichero pre-procesado: {bg_path_processed.name}")
        
        if not bg_path_processed.exists():
            raise FileNotFoundError(f"No se encontró un fichero de background para SHAP "
                                    f"(ni raw ni procesado) para el clasificador '{args.clf}' en el fold {args.fold}.")
        background_data = joblib.load(bg_path_processed)

    norm_params = joblib.load(fold_dir / 'vae_norm_params.joblib')
    label_info = _load_label_info(fold_dir)

    # ROI order (guardado a nivel run_dir en entrenamiento)
    roi_order_path_joblib = Path(args.run_dir) / 'roi_order_131.joblib'
    if roi_order_path_joblib.exists():
        roi_names = joblib.load(roi_order_path_joblib)
    else:
        # permitir pasarlo por CLI
        if args.roi_order_path is None:
            raise FileNotFoundError("No se encontró roi_order_131.joblib; especifícalo con --roi_order_path.")
        roi_names = _load_roi_names(Path(args.roi_order_path))

    # Datos globales ----------------------------------------------------------
    tensor_all, merged = _load_global_and_merge(Path(args.global_tensor_path), Path(args.metadata_path))
    cnad = _subset_cnad(merged)

    # Índices de test en cn_ad_df (order = usado al entrenar clasificador)
    test_idx_in_cnad = np.load(fold_dir / 'test_indices.npy')
    test_df = cnad.iloc[test_idx_in_cnad].copy()
    gidx_test = test_df['tensor_idx'].values

    # Reconstruir tensores test normalizados ----------------------------------
    tens_test = tensor_all[gidx_test][:, args.channels_to_use, :, :]
    tens_test = apply_normalization_params(tens_test, norm_params)
    tens_test_t = torch.from_numpy(tens_test).float()

    # Construir VAE y obtener latentes ----------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae_kwargs = _vae_kwargs_from_args(args, image_size=tensor_all.shape[-1])
    vae = build_vae(vae_kwargs, fold_dir / f"vae_model_fold_{args.fold}.pt", device)

    with torch.no_grad():
        recon, mu, logvar, z = vae(tens_test_t.to(device))  # forward completo
    lat_np = mu.cpu().numpy() if args.latent_features_type == 'mu' else z.cpu().numpy()
    lat_cols = [f'latent_{i}' for i in range(lat_np.shape[1])]
    X_lat = pd.DataFrame(lat_np, columns=lat_cols)

    # Features crudas = latentes + metadatos seleccionados --------------------
    meta_cols = args.metadata_features or []
    X_raw = pd.concat([X_lat.reset_index(drop=True),
                       test_df[meta_cols].reset_index(drop=True)], axis=1)
    log.info(f"[SHAP] X_raw shape={X_raw.shape} (latentes + {len(meta_cols)} metadatos)")

    # Transformar con preprocessor del pipeline --------------------------------
    preproc = pipe.named_steps['preprocessor']
    # Paso 1: preprocesado (one-hot, escalado…)
    X_proc = preproc.transform(X_raw)
    feat_names = preproc.get_feature_names_out()

    # Paso 2: SI el pipeline lleva selector de features, aplicarlo
    selector = pipe.named_steps.get("feature_select") \
               or pipe.named_steps.get("fs")            # según como lo hayas nombrado
    if selector is not None:
        X_proc = selector.transform(X_proc)
        # Ajustar los nombres de columnas al subconjunto seleccionado
        support_mask = selector.get_support()
        feat_names = feat_names[support_mask]

    # Construir DataFrame final coherente con el clasificador
    X_proc_df = pd.DataFrame(X_proc, columns=feat_names)

    # Determinar modelo a explicar --------------------------------------------
    model = unwrap_model_for_shap(pipe.named_steps['model'], args.clf)

    # Asegurar que background esté procesado con las mismas columnas ----------
    background_proc = _ensure_background_processed(background_data, preproc, feat_names, selector)

    # Construir explicador ----------------------------------------------------
    if args.clf in {'xgb', 'gb', 'rf', 'lgbm'}:
        explainer = shap.TreeExplainer(model, background_proc)
        shap_all = explainer.shap_values(X_proc_df)
    else:
        log.warning("[SHAP] Usando KernelExplainer (puede ser lento).")
        explainer = shap.KernelExplainer(model.predict_proba, background_proc)
        shap_all = explainer.shap_values(X_proc_df, nsamples=args.kernel_nsamples)

    # Seleccionar clase positiva ----------------------------------------------
    classes_ = list(model.classes_) if hasattr(model, 'classes_') else [0, 1]
    pos_int = label_info['positive_label_int']
    pos_idx = classes_.index(pos_int)
    shap_pos = _to_sample_feature(shap_all, pos_idx, *X_proc_df.shape)

    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[pos_idx]

    # Empaquetar artefactos ---------------------------------------------------
    pack = {
        'shap_values': shap_pos.astype(np.float32),
        'base_value': float(base_val),
        'X_test': X_proc_df,  # procesado
        'feature_names': feat_names.tolist(),
        'test_subject_ids': test_df['SubjectID'].astype(str).tolist(),
        'test_labels': test_df['ResearchGroup_Mapped'].map({'CN': 0, 'AD': 1}).astype(int).tolist(),
        'latent_features_type': args.latent_features_type,
        'metadata_features': meta_cols,
    }
    pack_path = out_dir / f'shap_pack_{args.clf}.joblib'
    joblib.dump(pack, pack_path)
    log.info(f"[SHAP] Pack guardado: {pack_path}")

    # Plots rápidos ------------------------------------------------------------
    _plot_shap_summary(shap_pos, X_proc_df, out_dir, args.fold, args.clf, base_val)



# ---------------------------------------------------------------------------
# SALIENCIA (subcomando "saliency")
# ---------------------------------------------------------------------------

def get_latent_weights_from_pack(pack: Dict[str, Any], mode: str, top_k: Optional[int]) -> pd.DataFrame:
    """Calcula pesos de las features latentes a partir de un shap_pack.

    mode:
        * mean_abs         → media de |SHAP| en todos los sujetos.
        * mean_signed      → media signed en todos los sujetos.
        * ad_vs_cn_diff    → (media SHAP AD) − (media SHAP CN)  (recomendado).
    """
    shap_values = pack['shap_values']            # (N, F)
    feature_names = pack['feature_names']        # list len=F
    labels = np.asarray(pack['test_labels'])     # (N,)

    import re
    latent_mask = np.array([bool(re.search(r'(?:^|__)latent_\d+$', n)) for n in feature_names])

    latent_vals = shap_values[:, latent_mask]
    latent_names = np.array(feature_names)[latent_mask]

    if mode == 'mean_abs':
        importance = np.abs(latent_vals).mean(axis=0)
    elif mode == 'mean_signed':
        importance = latent_vals.mean(axis=0)
    elif mode == 'ad_vs_cn_diff':
        imp_ad = latent_vals[labels == 1].mean(axis=0)
        imp_cn = latent_vals[labels == 0].mean(axis=0)
        importance = imp_ad - imp_cn
    else:
        raise ValueError(f"Modo de pesos SHAP no reconocido: {mode}")

    df = pd.DataFrame({'feature': latent_names, 'importance': importance})
    df['latent_idx'] = (
        df['feature']
        .str.extract(r'(\d+)$', expand=False)
        .astype(int)
    )

    # ordenar por magnitud absoluta (para top_k)
    df = df.reindex(df['importance'].abs().sort_values(ascending=False).index)
    if top_k is not None and top_k > 0:
        df = df.head(min(top_k, len(df)))
    # pesos normalizados (usar magnitud absoluta para normalizar; conservar signo en importance si te interesa)
    denom = df['importance'].abs().sum()
    df['weight'] = 0.0 if denom == 0 else df['importance'] / denom
    return df[['latent_idx', 'weight', 'importance', 'feature']]


def _vae_kwargs_from_args(args: argparse.Namespace, image_size: int) -> Dict[str, Any]:
    return dict(
        input_channels=len(args.channels_to_use),
        latent_dim=args.latent_dim,
        image_size=image_size,
        dropout_rate=args.dropout_rate_vae,
        use_layernorm_fc=getattr(args, 'use_layernorm_vae_fc', False),
        num_conv_layers_encoder=args.num_conv_layers_encoder,
        decoder_type=args.decoder_type,
        intermediate_fc_dim_config=args.intermediate_fc_dim_vae,
        final_activation=args.vae_final_activation,
        num_groups=args.gn_num_groups,
    )


def _load_roi_names(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == '.joblib':
        return joblib.load(path)
    if path.suffix == '.npy':
        return np.load(path, allow_pickle=True).astype(str).tolist()
    # txt/csv
    return pd.read_csv(path, header=None).iloc[:, 0].astype(str).tolist()


def generate_saliency_vectorized(vae_model: ConvolutionalVAE,
                                 weights_df: pd.DataFrame,
                                 input_tensor: torch.Tensor,
                                 device: torch.device) -> np.ndarray:
    """Genera mapa de saliencia promediando gradientes absolutos sobre batch.

    Devuelve array (C, R, R) en CPU numpy.
    """
    if input_tensor.numel() == 0:
        return np.zeros((vae_model.input_channels, vae_model.image_size, vae_model.image_size), dtype=np.float32)

    vae_model.eval()
    x = input_tensor.clone().detach().to(device)
    x.requires_grad = True

    # vector de pesos latentes (1, latent_dim)
    w = torch.zeros((1, vae_model.latent_dim), device=device, dtype=torch.float32)
    idx = torch.as_tensor(weights_df['latent_idx'].values, device=device, dtype=torch.long)
    vals = torch.as_tensor(weights_df['weight'].values, device=device, dtype=torch.float32)
    w[0, idx] = vals
    w = w.repeat(x.shape[0], 1)  # expand a batch

    vae_model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        # asumiendo forward encode() devuelve (mu, logvar)
        mu, _ = vae_model.encode(x)  # shape (B, latent_dim)
        mu.backward(gradient=w)

    sal = x.grad.detach().abs().mean(dim=0).cpu().numpy()  # (C,R,R)

    # Forzar simetría & diag=0 para cada canal
    sal_sym = 0.5 * (sal + sal.transpose(0, 2, 1))
    for c in range(sal_sym.shape[0]):
        np.fill_diagonal(sal_sym[c], 0.0)
    return sal_sym.astype(np.float32)


def _ensure_background_processed(
        background_data: Any,
        preproc: Any,
        feat_names_target: Sequence[str],
        selector: Optional[Any] = None
) -> pd.DataFrame:

    if isinstance(background_data, pd.DataFrame):
        # 1️⃣  Caso ideal: ya vienen con los nombres correctos
        if list(background_data.columns) == list(feat_names_target):
            return background_data

        # 2️⃣  Mismo nº de columnas ⇒ asumimos que YA está procesado,
        #     solo le ponemos los nombres que espera el clasificador
        if background_data.shape[1] == len(feat_names_target):
            log.info("[SHAP] Background ya parece procesado; renombrando columnas.")
            return background_data.set_axis(feat_names_target, axis=1, copy=False)

        # 3️⃣  Si no, intentamos procesarlo desde cero
        log.info("[SHAP] Background DataFrame detectado pero columnas no coinciden; transformando…")
        X_proc = preproc.transform(background_data)
        if selector is not None:
            X_proc = selector.transform(X_proc)
        return pd.DataFrame(X_proc, columns=feat_names_target)
    # -----------------------------------------------------------------
    # resto del cuerpo idéntico, pero añade selector en el branch ndarray
    if hasattr(background_data, "shape"):
        if background_data.shape[1] != len(feat_names_target):
            log.warning("[SHAP] Background ndarray con #cols distinto; transformando…")
            X_proc = preproc.transform(background_data)
            if selector is not None:
                X_proc = selector.transform(X_proc)
            return pd.DataFrame(X_proc, columns=feat_names_target)
        return background_data
    # -----------------------------------------------------------------
    raise TypeError(f"Tipo de background desconocido: {type(background_data)}")



def _plot_shap_summary(shap_pos: np.ndarray,
                       X_proc_df: pd.DataFrame,
                       out_dir: Path,
                       fold: int,
                       clf: str,
                       base_val: float) -> None:
    # bar
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_pos, X_proc_df, plot_type='bar', show=False, max_display=20)
    plt.title(f'SHAP Importancia Global (bar) - Fold {fold} - {clf.upper()}')
    plt.tight_layout()
    plt.savefig(out_dir / 'shap_global_importance_bar.png', dpi=150)
    plt.close()

    # beeswarm
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_pos, X_proc_df, show=False, max_display=20)
    plt.title(f'SHAP Impacto Features (beeswarm) - Fold {fold} - {clf.upper()}')
    plt.tight_layout()
    plt.savefig(out_dir / 'shap_summary_beeswarm.png', dpi=150)
    plt.close()

    # waterfall (primer sujeto) --- opcional: proteger si no hay muestras
    if shap_pos.shape[0] > 0:
        exp = shap.Explanation(
            values=shap_pos,
            base_values=np.full(shap_pos.shape[0], base_val, dtype=float),
            data=X_proc_df.values,
            feature_names=list(X_proc_df.columns)
        )
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(exp[0], max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(out_dir / 'shap_waterfall_subject_0.png', dpi=150)
        plt.close()



def cmd_saliency(args: argparse.Namespace) -> None:
    fold_dir = Path(args.run_dir) / f"fold_{args.fold}"
    shap_dir = fold_dir / 'interpretability_shap'
    pack_path = shap_dir / f'shap_pack_{args.clf}.joblib'
    if not pack_path.exists():
        raise FileNotFoundError(f"No se encontró shap_pack para {args.clf} en {pack_path}. Corre primero el subcomando 'shap'.")

    pack = joblib.load(pack_path)
    log.info(f"[SALIENCY] fold={args.fold} clf={args.clf}  (pack cargado: {pack_path.name})")
    # ------------------------------------------------------------------
    roi_names: Optional[List[str]] = None
    roi_order_joblib = Path(args.run_dir) / 'roi_order_131.joblib'
    if roi_order_joblib.exists():
        roi_names = joblib.load(roi_order_joblib)
        log.info(f"Usando ROI order de {roi_order_joblib}.")
    elif getattr(args, "roi_order_path", None):
        p_roi = Path(args.roi_order_path)
        roi_names = _load_roi_names(p_roi)
        log.info(f"Usando ROI order de --roi_order_path={p_roi}.")

    # Intento con anotaciones si aún no tenemos roi_names
    roi_map_df: Optional[pd.DataFrame] = None
    annot_path = args.roi_annotation_path or args.roi_annotation_csv
    if annot_path is not None:
        try:
            roi_map_df = pd.read_csv(annot_path)
            log.info(f"Cargado fichero de anotaciones: {annot_path}")
            if roi_names is None:
                # ordenar por ROI_TensorIdx si existe; si no, por Original_Index_0_N
                if "ROI_TensorIdx" in roi_map_df.columns:
                    roi_map_df = roi_map_df.sort_values("ROI_TensorIdx").reset_index(drop=True)
                elif "Original_Index_0_N" in roi_map_df.columns:
                    roi_map_df = roi_map_df.sort_values("Original_Index_0_N").reset_index(drop=True)
                roi_names = roi_map_df["AAL3_Name"].astype(str).tolist()
                log.info("Derivado orden de ROIs desde el CSV de anotaciones.")
        except FileNotFoundError:
            log.error(f"No se pudo leer el fichero de anotaciones: {annot_path}")

    if roi_names is None:
        raise FileNotFoundError(
            "No pude resolver el orden de ROIs. Proporciona --roi_order_path "
            "o asegúrate de tener roi_order_131.joblib o un CSV anotado con AAL3_Name."
        )


    # Pesos latentes ----------------------------------------------------------
    weights_df = get_latent_weights_from_pack(pack, args.shap_weight_mode, args.top_k)
    log.info(f"[SALIENCY] {len(weights_df)} latentes ponderadas. Ejemplo:\n{weights_df.head().to_string(index=False)}")

    # Datos test en espacio de entrada original --------------------------------
    tensor_all, merged = _load_global_and_merge(Path(args.global_tensor_path), Path(args.metadata_path))
    cnad = _subset_cnad(merged)
    test_idx_in_cnad = np.load(fold_dir / 'test_indices.npy')
    test_df = cnad.iloc[test_idx_in_cnad].copy()
    gidx_test = test_df['tensor_idx'].values

    norm_params = joblib.load(fold_dir / 'vae_norm_params.joblib')
    tens_test = tensor_all[gidx_test][:, args.channels_to_use, :, :]
    tens_test = apply_normalization_params(tens_test, norm_params)
    tens_test_t = torch.from_numpy(tens_test).float()

    n_rois_tensor = tens_test.shape[-1]
    if len(roi_names) != n_rois_tensor:
        log.error(f"Número de ROIs ({len(roi_names)}) != dimensión tensor ({n_rois_tensor}). "
                  "Verifica que el orden de ROIs corresponde al tensor usado en entrenamiento.")
        raise ValueError("Desajuste longitud roi_names vs tensor.")

    # VAE ----------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae_kwargs = _vae_kwargs_from_args(args, image_size=tens_test.shape[-1])
    vae = build_vae(vae_kwargs, fold_dir / f"vae_model_fold_{args.fold}.pt", device)

    labels = np.asarray(pack['test_labels'])  # CN=0 / AD=1 según pack
    x_ad = tens_test_t[labels == 1]
    x_cn = tens_test_t[labels == 0]

    log.info(f"[SALIENCY] Sujetos AD={x_ad.shape[0]}  CN={x_cn.shape[0]}")

    sal_ad = generate_saliency_vectorized(vae, weights_df, x_ad, device)
    sal_cn = generate_saliency_vectorized(vae, weights_df, x_cn, device)
    sal_diff = sal_ad - sal_cn

    # Guardar mapas ------------------------------------------------------------
    out_dir = fold_dir / f"interpretability_{args.clf}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"saliency_map_ad_top{args.top_k}.npy", sal_ad)
    np.save(out_dir / f"saliency_map_cn_top{args.top_k}.npy", sal_cn)
    np.save(out_dir / f"saliency_map_diff_top{args.top_k}.npy", sal_diff)

    _ranking_and_heatmap(
        saliency_map_diff=sal_diff,
        roi_map_df=roi_map_df,
        roi_names=roi_names,
        out_dir=out_dir,
        fold=args.fold,
        clf=args.clf,
        top_k=args.top_k
    )

 
    # Guardar args usados ------------------------------------------------------
    with open(out_dir / f"run_args_saliency_top{args.top_k}.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

    log.info(f"[SALIENCY] Completo. Resultados en {out_dir}")


# ---------------------------------------------------------------------------
# Ranking + visualización
# ---------------------------------------------------------------------------

def _ranking_and_heatmap(saliency_map_diff: np.ndarray,
                         roi_map_df: Optional[pd.DataFrame],
                         roi_names: Sequence[str],
                         out_dir: Path,
                         fold: int,
                         clf: str,
                         top_k: int,
                         annot_df: Optional[pd.DataFrame] = None) -> None:
    # saliency_map_diff: (C,R,R) → promediamos sobre canales
    sal_m = saliency_map_diff.mean(axis=0)
    n_rois = sal_m.shape[0]
    # Crear la tabla de conexiones con índices numéricos
    ut_indices = np.triu_indices(n_rois, k=1)
    df_edges = pd.DataFrame({
        'idx_i': ut_indices[0],
        'idx_j': ut_indices[1],
        'Saliency_Score': sal_m[ut_indices]
    })
    # <NUEVO> Anotar el DataFrame de conexiones si el mapa de ROIs está disponible
    if roi_map_df is not None:
        # Alinear al orden del tensor
        if 'ROI_TensorIdx' in roi_map_df.columns:
            roi_map_df = roi_map_df.sort_values('ROI_TensorIdx').reset_index(drop=True)
            idx_col = 'ROI_TensorIdx'
        elif 'Original_Index_0_N' in roi_map_df.columns:
            roi_map_df = roi_map_df.sort_values('Original_Index_0_N').reset_index(drop=True)
            idx_col = 'Original_Index_0_N'
        else:
            log.warning("roi_map_df no tiene columnas de índice reconocibles; se omitirá la anotación.")
            idx_col = None

        if idx_col is not None:
            # Diccionarios de anotación
            name_map   = roi_map_df.set_index(idx_col)['AAL3_Name'].astype(str).to_dict()
            lobe_map   = roi_map_df.set_index(idx_col)['Macro_Lobe'].astype(str).to_dict()           if 'Macro_Lobe' in roi_map_df else {}
            net_map    = roi_map_df.set_index(idx_col)['Refined_Network'].astype(str).to_dict()      if 'Refined_Network' in roi_map_df else {}
            y17_map    = roi_map_df.set_index(idx_col)['Yeo17_Network'].astype(str).to_dict()        if 'Yeo17_Network' in roi_map_df else {}

            # Añadir columnas al ranking
            df_edges['ROI_i_name'] = df_edges['idx_i'].map(name_map)
            df_edges['ROI_j_name'] = df_edges['idx_j'].map(name_map)
            df_edges['src_AAL3_Name'] = df_edges['ROI_i_name']
            df_edges['dst_AAL3_Name'] = df_edges['ROI_j_name']
            df_edges['src_Macro_Lobe'] = df_edges['idx_i'].map(lobe_map)
            df_edges['dst_Macro_Lobe'] = df_edges['idx_j'].map(lobe_map)
            df_edges['src_Refined_Network'] = df_edges['idx_i'].map(net_map)
            df_edges['dst_Refined_Network'] = df_edges['idx_j'].map(net_map)
            df_edges['src_Yeo17_Network'] = df_edges['idx_i'].map(y17_map)
            df_edges['dst_Yeo17_Network'] = df_edges['idx_j'].map(y17_map)      


    df_edges['Saliency_Magnitude'] = df_edges['Saliency_Score'].abs()
    df_edges = df_edges.sort_values('Saliency_Magnitude', ascending=False).drop(columns='Saliency_Magnitude')
    df_edges.insert(0, 'Rank', range(1, len(df_edges) + 1))
    edge_csv_path = out_dir / f"ranking_conexiones_ANOTADO_top{top_k}.csv"
    df_edges.to_csv(edge_csv_path, index=False)
    log.info(f"[SALIENCY] Ranking de conexiones ANOTADO guardado: {edge_csv_path}")

    if annot_df is not None:
        meta = annot_df[['AAL3_Name','Macro_Lobe','Refined_Network']]
        df_edges = (df_edges
                    .merge(meta, left_on='ROI_i_name', right_on='AAL3_Name', how='left')
                    .rename(columns={'Macro_Lobe':'Lobe_i','Refined_Network':'Network_i'})
                    .drop(columns='AAL3_Name')
                    .merge(meta, left_on='ROI_j_name', right_on='AAL3_Name', how='left')
                    .rename(columns={'Macro_Lobe':'Lobe_j','Refined_Network':'Network_j'})
                    .drop(columns='AAL3_Name'))
    # preview top 20
    preview_cols = ['Rank', 'src_AAL3_Name', 'dst_AAL3_Name', 'Saliency_Score', 'src_Refined_Network', 'dst_Refined_Network']
    preview_cols_exist = [c for c in preview_cols if c in df_edges.columns]
    log.info("Top 20 conexiones anotadas:\n" + df_edges.head(20)[preview_cols_exist].to_string())
 

    # heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(sal_m, cmap='coolwarm', center=0,
                xticklabels=list(roi_names)[:n_rois], yticklabels=list(roi_names)[:n_rois],
                cbar_kws={'label': 'Saliencia Diferencial (AD > CN)'} )
    plt.title(f'Mapa de Saliencia Diferencial (AD vs CN) - Fold {fold} - {clf.upper()}')
    plt.tight_layout(); plt.savefig(out_dir / f"mapa_saliencia_diferencial_top{top_k}.png", dpi=150); plt.close()


# ---------------------------------------------------------------------------
# Argumentos CLI
# ---------------------------------------------------------------------------

def _add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--roi_annotation_csv', default=None,
               help='CSV con anotaciones de ROI (Macro_Lobe, Redes, etc.).')
    p.add_argument('--run_dir', required=True, help='Directorio raíz del experimento (donde viven fold_*).')
    p.add_argument('--fold', type=int, required=True, help='Fold a analizar (1-indexed).')
    p.add_argument('--clf', required=True, help='Clasificador (xgb, svm, logreg, gb, rf, ...).')
    p.add_argument('--global_tensor_path', required=True, help='Ruta al GLOBAL_TENSOR .npz usado en entrenamiento.')
    p.add_argument('--metadata_path', required=True, help='Ruta al CSV de metadatos usado en entrenamiento.')
    p.add_argument('--channels_to_use', type=int, nargs='*', required=True, help='Índices de canales usados en entrenamiento.')
    p.add_argument('--latent_dim', type=int, required=True, help='Dimensión latente del VAE.')
    p.add_argument('--latent_features_type', choices=['mu','z'], default='mu', help='Usar mu o z como features latentes.')
    p.add_argument('--metadata_features', nargs='*', default=None, help='Columnas de metadatos añadidas al clasificador.')
    # Arquitectura VAE
    p.add_argument('--num_conv_layers_encoder', type=int, default=4)
    p.add_argument('--decoder_type', default='convtranspose', choices=['convtranspose','upsample_conv'])
    p.add_argument('--dropout_rate_vae', type=float, default=0.2)
    p.add_argument('--use_layernorm_vae_fc', action='store_true')
    p.add_argument('--intermediate_fc_dim_vae', default='quarter')
    p.add_argument('--vae_final_activation', default='tanh', choices=['tanh','sigmoid','linear'])
    p.add_argument('--gn_num_groups', type=int, default=16, help='n grupos para GroupNorm en VAE.')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Pipeline Unificado de Interpretabilidad (VAE+Clasificador).',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)

    # subcomando SHAP ---------------------------------------------------------
    p_shap = sub.add_parser('shap', help='Calcular y guardar valores SHAP para un fold+clf.')
    _add_shared_args(p_shap)
    p_shap.add_argument('--roi_order_path', default=None, help='(Opcional) ruta a ROI order si no está en run_dir.')
    p_shap.add_argument('--kernel_nsamples', type=int, default=100, help='nsamples para KernelExplainer (modelos no tree).')

    # subcomando SALIENCY -----------------------------------------------------
    p_sal = sub.add_parser('saliency', help='Generar mapas de saliencia a partir del shap_pack.')
    _add_shared_args(p_sal)
    p_sal.add_argument('--roi_order_path', default=None,
                    help='(Opcional) ruta a ROI order (.joblib/.npy/.txt) si no está en run_dir.')
    p_sal.add_argument('--roi_annotation_path', required=True, help='Ruta al fichero maestro de anotaciones de ROIs (roi_info_master.csv).')
    p_sal.add_argument('--top_k', type=int, default=50, help='Nº máx features latentes a usar.')
    p_sal.add_argument('--shap_weight_mode', default='ad_vs_cn_diff', choices=['mean_abs','mean_signed','ad_vs_cn_diff'],
                       help='Cómo convertir valores SHAP latentes en pesos para saliencia.')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.cmd == 'shap':
        cmd_shap(args)
    elif args.cmd == 'saliency':
        cmd_saliency(args)
    else:
        raise ValueError(f"Subcomando desconocido: {args.cmd}")


if __name__ == '__main__':
    main()
