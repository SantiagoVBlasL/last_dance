#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
serentipia.py

Script para entrenar un Autoencoder Variacional (VAE) Convolucional sobre matrices de 
conectividad funcional fMRI y luego usar sus representaciones latentes para clasificar 
entre Controles Sanos (CN) y pacientes con Enfermedad de Alzheimer (AD).

Versión: 1.7.0 - Pipeline de Clasificador Unificado y Robusto
Cambios desde v1.6.0:
- Corregida la inconsistencia de pre-procesado (data leakage) al encapsular
  el escalado, SMOTE y el clasificador en un único `imblearn.pipeline.Pipeline`.
- El pre-procesado del clasificador (scaling, SMOTE) se elimina de la lógica
  del bucle principal y se delega completamente al pipeline obtenido de `classifiers.py`.
- Simplificado drásticamente el bucle de entrenamiento y predicción del clasificador,
  eliminando la gestión manual de `StandardScaler` y `SMOTE`.
- `RandomizedSearchCV` ahora opera sobre el pipeline completo, asegurando que
  todas las transformaciones se apliquen de forma consistente.
- El modelo final entrenado (`final_clf_model`) es el pipeline completo, garantizando
  que en la predicción se aplique la misma cadena de transformaciones.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("sklearn").setLevel(logging.ERROR)
logging.getLogger("lightgbm").setLevel(logging.ERROR)
from models.convolutional_vae_new import ConvolutionalVAE
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
import argparse
import gc
import copy
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*X does not have valid feature names.*",
    category=UserWarning,
    module="sklearn.utils.validation"
)
import subprocess
from models.classifiers import get_classifier_and_grid, get_available_classifiers
import joblib
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split as sk_train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, average_precision_score, balanced_accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# --- Configuración del Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# --- Constantes y Configuraciones Globales ---
DEFAULT_CHANNEL_NAMES = [
    'Pearson_OMST_GCE_Signed_Weighted', 'Pearson_Full_FisherZ_Signed', 'MI_KNN_Symmetric',
    'dFC_AbsDiffMean', 'dFC_StdDev', 'Granger_F_lag1'
]

# ... (Las funciones auxiliares como vae_loss_function, load_data, etc. no cambian) ...
def vae_loss_function(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    recon_loss_mse = nn.functional.mse_loss(recon_x, x, reduction='sum') / x.shape[0]
    kld_element = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kld = torch.mean(kld_element)
    return recon_loss_mse + beta * kld

def get_cyclical_beta_schedule(current_epoch: int, total_epochs: int, beta_max: float, n_cycles: int, ratio_increase: float = 0.5) -> float:
    if n_cycles <= 0: return beta_max
    epoch_per_cycle = total_epochs / n_cycles
    epoch_in_current_cycle = current_epoch % epoch_per_cycle
    increase_phase_duration = epoch_per_cycle * ratio_increase
    return beta_max * (epoch_in_current_cycle / increase_phase_duration) if epoch_in_current_cycle < increase_phase_duration else beta_max

def load_data(tensor_path: Path, metadata_path: Path) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
    logger.info(f"Cargando tensor global desde: {tensor_path}")
    if not tensor_path.exists():
        logger.error(f"Archivo de tensor global NO encontrado: {tensor_path}")
        return None, None
    try:
        data_npz = np.load(tensor_path)
        global_tensor = data_npz['global_tensor_data']
        subject_ids_tensor = data_npz['subject_ids'].astype(str) 
        logger.info(f"Tensor global cargado. Forma: {global_tensor.shape}")
    except Exception as e:
        logger.error(f"Error cargando tensor global: {e}")
        return None, None

    logger.info(f"Cargando metadatos desde: {metadata_path}")
    if not metadata_path.exists():
        logger.error(f"Archivo de metadatos NO encontrado: {metadata_path}")
        return None, None
    try:
        metadata_df = pd.read_csv(metadata_path)
        metadata_df['SubjectID'] = metadata_df['SubjectID'].astype(str).str.strip()
        logger.info(f"Metadatos cargados. Forma: {metadata_df.shape}")
    except Exception as e:
        logger.error(f"Error cargando metadatos: {e}")
        return None, None

    tensor_df = pd.DataFrame({'SubjectID': subject_ids_tensor})
    tensor_df['tensor_idx'] = np.arange(len(subject_ids_tensor))
    merged_df = pd.merge(tensor_df, metadata_df, on='SubjectID', how='left')
    
    num_tensor_subjects = len(subject_ids_tensor)
    if len(merged_df) < num_tensor_subjects:
         logger.warning(f"Algunos SubjectIDs del tensor ({num_tensor_subjects}) no se encontraron en los metadatos. Merged: {len(merged_df)}.")
    return global_tensor, merged_df

def normalize_inter_channel_fold(
    data_tensor: np.ndarray, 
    train_indices_in_fold: np.ndarray, 
    mode: str = 'zscore_offdiag',
    selected_channel_original_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    # Esta función no cambia, su lógica para normalizar el tensor VAE es correcta
    # y sigue operando sobre los datos de entrenamiento del VAE de cada fold.
    num_subjects_total, num_selected_channels, num_rois, _ = data_tensor.shape
    logger.info(f"Aplicando normalización inter-canal (modo: {mode}) sobre {num_selected_channels} canales seleccionados.")
    logger.info(f"Parámetros de normalización se calcularán usando {len(train_indices_in_fold)} sujetos de entrenamiento.")
    
    normalized_tensor_fold = data_tensor.copy()
    norm_params_per_channel_list = []
    off_diag_mask = ~np.eye(num_rois, dtype=bool)

    for c_idx_selected in range(num_selected_channels):
        current_channel_original_name = selected_channel_original_names[c_idx_selected] if selected_channel_original_names and c_idx_selected < len(selected_channel_original_names) else f"Channel_{c_idx_selected}"
        params = {'mode': mode, 'original_name': current_channel_original_name}
        use_fixed_params = False

        if mode == 'minmax_offdiag' and current_channel_original_name in FIXED_MINMAX_PARAMS_PER_CHANNEL:
            fixed_p = FIXED_MINMAX_PARAMS_PER_CHANNEL[current_channel_original_name]
            params.update({'min': fixed_p['min'], 'max': fixed_p['max']})
            use_fixed_params = True
            logger.info(f"Canal '{current_channel_original_name}': Usando MinMax fijo (min={params['min']:.4f}, max={params['max']:.4f}).")

        if not use_fixed_params:
            channel_data_train_for_norm_params = data_tensor[train_indices_in_fold, c_idx_selected, :, :]
            all_off_diag_train_values = channel_data_train_for_norm_params[:, off_diag_mask].flatten()

            if all_off_diag_train_values.size == 0:
                logger.warning(f"Canal '{current_channel_original_name}': No hay elementos fuera de la diagonal en el training set. No se escala.")
                params.update({'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 1.0, 'no_scale': True})
            elif mode == 'zscore_offdiag':
                mean_val = np.mean(all_off_diag_train_values)
                std_val = np.std(all_off_diag_train_values)
                params.update({'mean': mean_val, 'std': std_val if std_val > 1e-9 else 1.0})
                if std_val <= 1e-9: logger.warning(f"Canal '{current_channel_original_name}': STD muy bajo ({std_val:.2e}). Usando STD=1.")
            elif mode == 'minmax_offdiag':
                min_val = np.min(all_off_diag_train_values)
                max_val = np.max(all_off_diag_train_values)
                params.update({'min': min_val, 'max': max_val})
                if (max_val - min_val) <= 1e-9: logger.warning(f"Canal '{current_channel_original_name}': Rango (max-min) muy bajo ({(max_val - min_val):.2e}).")
            else:
                raise ValueError(f"Modo de normalización desconocido: {mode}")
        
        norm_params_per_channel_list.append(params)

        if not params.get('no_scale', False):
            current_channel_data_all_subjects = data_tensor[:, c_idx_selected, :, :]
            scaled_channel_data = current_channel_data_all_subjects.copy()
            if off_diag_mask.any():
                if mode == 'zscore_offdiag':
                    if params['std'] > 1e-9:
                        scaled_channel_data[:, off_diag_mask] = (current_channel_data_all_subjects[:, off_diag_mask] - params['mean']) / params['std']
                elif mode == 'minmax_offdiag':
                    range_val = params.get('max', 1.0) - params.get('min', 0.0)
                    if range_val > 1e-9: 
                        scaled_channel_data[:, off_diag_mask] = (current_channel_data_all_subjects[:, off_diag_mask] - params['min']) / range_val
                    else: 
                        scaled_channel_data[:, off_diag_mask] = 0.0 
            normalized_tensor_fold[:, c_idx_selected, :, :] = scaled_channel_data
            if not use_fixed_params:
                log_msg_params = f"Canal '{current_channel_original_name}': Off-diag {mode} (train_params: "
                if mode == 'zscore_offdiag': log_msg_params += f"mean={params['mean']:.3f}, std={params['std']:.3f})"
                elif mode == 'minmax_offdiag': log_msg_params += f"min={params['min']:.3f}, max={params['max']:.3f})"
                logger.info(log_msg_params)
    return normalized_tensor_fold, norm_params_per_channel_list


def apply_normalization_params(data_tensor_subset: np.ndarray, 
                               norm_params_per_channel_list: List[Dict[str, float]]
                               ) -> np.ndarray:
    # Esta función tampoco cambia
    num_subjects, num_selected_channels, num_rois, _ = data_tensor_subset.shape
    normalized_tensor_subset = data_tensor_subset.copy()
    off_diag_mask = ~np.eye(num_rois, dtype=bool)

    if len(norm_params_per_channel_list) != num_selected_channels:
        raise ValueError(f"Mismatch in number of channels for normalization: data has {num_selected_channels}, params provided for {len(norm_params_per_channel_list)}")

    for c_idx_selected in range(num_selected_channels):
        params = norm_params_per_channel_list[c_idx_selected]
        mode = params.get('mode', 'zscore_offdiag') 
        if params.get('no_scale', False):
            continue
        current_channel_data = data_tensor_subset[:, c_idx_selected, :, :]
        scaled_channel_data_subset = current_channel_data.copy()
        if off_diag_mask.any():
            if mode == 'zscore_offdiag':
                if params['std'] > 1e-9:
                    scaled_channel_data_subset[:, off_diag_mask] = (current_channel_data[:, off_diag_mask] - params['mean']) / params['std']
            elif mode == 'minmax_offdiag':
                range_val = params.get('max', 1.0) - params.get('min', 0.0)
                if range_val > 1e-9:
                    scaled_channel_data_subset[:, off_diag_mask] = (current_channel_data[:, off_diag_mask] - params['min']) / range_val
                else:
                    scaled_channel_data_subset[:, off_diag_mask] = 0.0 
        normalized_tensor_subset[:, c_idx_selected, :, :] = scaled_channel_data_subset
    return normalized_tensor_subset

def log_group_distributions(df: pd.DataFrame, group_cols: List[str], dataset_name: str, fold_idx_str: str):
    # Sin cambios en esta función
    if df.empty:
        logger.info(f"  {fold_idx_str} {dataset_name}: DataFrame vacío.")
        return
    log_msg = f"  {fold_idx_str} {dataset_name} (N={len(df)}):\n"
    for col in group_cols:
        if col in df.columns:
            counts = df[col].value_counts().sort_index()
            log_msg += f"    {col}:\n"
            for val, count in counts.items():
                log_msg += f"      {val}: {count} ({count/len(df)*100:.1f}%)\n"
        else:
            log_msg += f"    {col}: No encontrado en el DataFrame.\n"
    logger.info(log_msg.strip())


def train_and_evaluate_pipeline(global_tensor_all_channels: np.ndarray, 
                                metadata_df_full: pd.DataFrame,
                                args: argparse.Namespace):
    
    # ... (la configuración inicial del pipeline y del VAE no cambia) ...
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    selected_channel_indices: List[int] = []
    selected_channel_names_in_tensor: List[str] = []
    if args.channels_to_use:
        for ch_specifier in args.channels_to_use:
            try:
                ch_idx = int(ch_specifier)
                if 0 <= ch_idx < global_tensor_all_channels.shape[1]:
                    selected_channel_indices.append(ch_idx)
                else:
                    logger.warning(f"Índice de canal '{ch_idx}' fuera de rango. Ignorando.")
            except ValueError:
                if ch_specifier in DEFAULT_CHANNEL_NAMES:
                    try:
                        original_idx = DEFAULT_CHANNEL_NAMES.index(ch_specifier)
                        if original_idx < global_tensor_all_channels.shape[1]:
                            selected_channel_indices.append(original_idx)
                        else:
                             logger.warning(f"Nombre de canal '{ch_specifier}' (índice {original_idx}) fuera de rango. Ignorando.")
                    except ValueError: pass 
                else:
                    logger.warning(f"Nombre de canal '{ch_specifier}' desconocido. Ignorando.")
        
        if not selected_channel_indices:
            logger.error("No se seleccionaron canales válidos. Abortando.")
            return
        selected_channel_indices = sorted(list(set(selected_channel_indices)))
        temp_names = []
        for idx in selected_channel_indices:
            if idx < len(DEFAULT_CHANNEL_NAMES): 
                temp_names.append(DEFAULT_CHANNEL_NAMES[idx])
            elif idx < global_tensor_all_channels.shape[1]:
                temp_names.append(f"RawChan{idx}")
        selected_channel_names_in_tensor = temp_names
        current_global_tensor = global_tensor_all_channels[:, selected_channel_indices, :, :]
        logger.info(f"Usando canales seleccionados (índices): {selected_channel_indices}")
        logger.info(f"Nombres de canales seleccionados: {selected_channel_names_in_tensor}")
    else:
        current_global_tensor = global_tensor_all_channels
        selected_channel_names_in_tensor = [DEFAULT_CHANNEL_NAMES[i] if i < len(DEFAULT_CHANNEL_NAMES) else f"RawChan{i}" for i in range(current_global_tensor.shape[1])]
        logger.info(f"Usando todos los {current_global_tensor.shape[1]} canales.")
    
    num_input_channels_for_vae = current_global_tensor.shape[1]

    if 'ResearchGroup' not in metadata_df_full.columns:
        logger.error("'ResearchGroup' no encontrado. Abortando.")
        return
    cn_ad_df = metadata_df_full[metadata_df_full['ResearchGroup'].isin(['CN', 'AD'])].copy()
    if cn_ad_df.empty or 'tensor_idx' not in cn_ad_df.columns:
        logger.error("No hay sujetos CN/AD o falta 'tensor_idx' en el DataFrame mergeado. Abortando.")
        return
    
    max_valid_idx_for_cn_ad = current_global_tensor.shape[0] - 1
    original_cn_ad_count = len(cn_ad_df)
    cn_ad_df = cn_ad_df[cn_ad_df['tensor_idx'] <= max_valid_idx_for_cn_ad].copy()
    if len(cn_ad_df) < original_cn_ad_count:
        logger.warning(f"Algunos sujetos CN/AD filtrados porque 'tensor_idx' excede las dimensiones del tensor. "
                       f"Original: {original_cn_ad_count}, Post-filtro: {len(cn_ad_df)}")

    if cn_ad_df.empty:
        logger.error("No hay sujetos CN/AD válidos después de filtrar por tensor_idx. Abortando.")
        return

    cn_ad_df['label'] = cn_ad_df['ResearchGroup'].map({'CN': 0, 'AD': 1})
    
    strat_cols = ['ResearchGroup']
    if args.classifier_stratify_cols:
        for col in args.classifier_stratify_cols:
            if col in cn_ad_df.columns:
                cn_ad_df[col] = cn_ad_df[col].fillna(f"{col}_Unknown").astype(str)
                if col not in strat_cols: strat_cols.append(col)
            else: logger.warning(f"Columna de estratificación '{col}' no encontrada.")
    
    cn_ad_df['stratify_key_clf'] = cn_ad_df[strat_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    logger.info(f"Estratificando folds del clasificador por: {strat_cols}")

    X_classifier_subject_indices_in_cn_ad_df = np.arange(len(cn_ad_df))
    y_classifier_labels_cn_ad = cn_ad_df['label'].values
    stratify_key_for_clf_cv = cn_ad_df['stratify_key_clf']
    
    logger.info(f"Sujetos CN/AD para clasificación: {len(cn_ad_df)}. CN: {sum(y_classifier_labels_cn_ad == 0)}, AD: {sum(y_classifier_labels_cn_ad == 1)}")

    if args.repeated_outer_folds_n_repeats > 1:
        outer_cv_clf = RepeatedStratifiedKFold(n_splits=args.outer_folds, n_repeats=args.repeated_outer_folds_n_repeats, random_state=args.seed)
        total_outer_iterations = args.outer_folds * args.repeated_outer_folds_n_repeats
    else:
        outer_cv_clf = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
        total_outer_iterations = args.outer_folds
    logger.info(f"Usando CV externa: {type(outer_cv_clf).__name__} con {total_outer_iterations} iteraciones totales.")

    all_folds_metrics = []
    all_folds_vae_history = []
    all_folds_clf_predictions = []

    for fold_idx, (train_dev_clf_idx_in_cn_ad_df, test_clf_idx_in_cn_ad_df) in enumerate(outer_cv_clf.split(X_classifier_subject_indices_in_cn_ad_df, stratify_key_for_clf_cv)):
        fold_start_time = time.time()
        fold_idx_str = f"Fold {fold_idx + 1}/{total_outer_iterations}"
        logger.info(f"--- Iniciando {fold_idx_str} ---")
        
        # ... (la lógica de entrenamiento del VAE hasta el guardado del modelo no cambia) ...
        fold_output_dir = Path(args.output_dir) / f"fold_{fold_idx + 1}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        global_indices_clf_test_this_fold = cn_ad_df.iloc[test_clf_idx_in_cn_ad_df]['tensor_idx'].values
        log_group_distributions(cn_ad_df.iloc[test_clf_idx_in_cn_ad_df], strat_cols, "Test Set (Clasificador)", fold_idx_str)

        all_valid_subject_indices_from_metadata = metadata_df_full[metadata_df_full['tensor_idx'] <= max_valid_idx_for_cn_ad]['tensor_idx'].values
        global_indices_vae_training_pool = np.setdiff1d(all_valid_subject_indices_from_metadata, global_indices_clf_test_this_fold, assume_unique=True)
        
        if len(global_indices_vae_training_pool) < 10: 
            logger.error(f"{fold_idx_str}: Muy pocos sujetos ({len(global_indices_vae_training_pool)}) para entrenamiento VAE. Saltando fold.")
            continue

        vae_train_pool_df = metadata_df_full[metadata_df_full['tensor_idx'].isin(global_indices_vae_training_pool)]
        log_group_distributions(vae_train_pool_df, ['ResearchGroup', 'Sex'], "Pool Entrenamiento VAE", fold_idx_str)
        
        vae_train_pool_tensor_original_scale = current_global_tensor[global_indices_vae_training_pool]
        
        vae_actual_train_indices_local_to_pool, vae_internal_val_indices_local_to_pool = [], []
        stratify_labels_vae_split = None
        if 'ResearchGroup' in vae_train_pool_df.columns:
            temp_vae_labels = vae_train_pool_df['ResearchGroup'].fillna('Unknown_Group').astype(str)
            if len(np.unique(temp_vae_labels)) > 1 and all(temp_vae_labels.value_counts() >= args.inner_folds if args.inner_folds > 1 else 2):
                stratify_labels_vae_split = temp_vae_labels
                logger.info(f"  {fold_idx_str} VAE val split será estratificado por ResearchGroup del pool VAE.")

        if args.vae_val_split_ratio > 0 and len(global_indices_vae_training_pool) > 10:
            try:
                vae_actual_train_indices_local_to_pool, vae_internal_val_indices_local_to_pool = sk_train_test_split(
                    np.arange(len(global_indices_vae_training_pool)), 
                    test_size=args.vae_val_split_ratio,
                    stratify=stratify_labels_vae_split,
                    random_state=args.seed + fold_idx + 10, shuffle=True 
                )
            except ValueError as e:
                logger.warning(f"  {fold_idx_str} No se pudo hacer split de validación VAE estratificado ({e}). Intentando split aleatorio.")
                vae_actual_train_indices_local_to_pool, vae_internal_val_indices_local_to_pool = sk_train_test_split(
                    np.arange(len(global_indices_vae_training_pool)), test_size=args.vae_val_split_ratio,
                    random_state=args.seed + fold_idx + 10, shuffle=True 
                )
        else:
            vae_actual_train_indices_local_to_pool = np.arange(len(global_indices_vae_training_pool))
        
        log_group_distributions(vae_train_pool_df.iloc[vae_actual_train_indices_local_to_pool], ['ResearchGroup', 'Sex'], "Actual Train Set (VAE)", fold_idx_str)
        if len(vae_internal_val_indices_local_to_pool) > 0:
             log_group_distributions(vae_train_pool_df.iloc[vae_internal_val_indices_local_to_pool], ['ResearchGroup', 'Sex'], "Internal Val Set (VAE)", fold_idx_str)
        logger.info(f"  {fold_idx_str} Sujetos VAE actual train: {len(vae_actual_train_indices_local_to_pool)}, VAE internal val: {len(vae_internal_val_indices_local_to_pool)}")

        vae_pool_tensor_norm, norm_params_fold_list = normalize_inter_channel_fold(
            vae_train_pool_tensor_original_scale, vae_actual_train_indices_local_to_pool, 
            mode=args.norm_mode, selected_channel_original_names=selected_channel_names_in_tensor
        )

        vae_train_dataset = TensorDataset(torch.from_numpy(vae_pool_tensor_norm[vae_actual_train_indices_local_to_pool]).float())
        vae_train_loader = DataLoader(vae_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        
        vae_internal_val_loader = None
        if len(vae_internal_val_indices_local_to_pool) > 0:
            vae_internal_val_dataset = TensorDataset(torch.from_numpy(vae_pool_tensor_norm[vae_internal_val_indices_local_to_pool]).float())
            vae_internal_val_loader = DataLoader(vae_internal_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"  {fold_idx_str} Usando dispositivo: {device}")
        
        vae_fold_k = ConvolutionalVAE(
            input_channels=num_input_channels_for_vae, latent_dim=args.latent_dim, image_size=current_global_tensor.shape[-1],
            final_activation=args.vae_final_activation, intermediate_fc_dim_config=args.intermediate_fc_dim_vae,
            dropout_rate=args.dropout_rate_vae, use_layernorm_fc=args.use_layernorm_vae_fc,
            num_conv_layers_encoder=args.num_conv_layers_encoder, decoder_type=args.decoder_type
        ).to(device)
        
        optimizer_vae = optim.Adam(vae_fold_k.parameters(), lr=args.lr_vae, weight_decay=args.weight_decay_vae)
        scheduler_vae = None
        if vae_internal_val_loader and args.lr_scheduler_patience_vae > 0:
            scheduler_vae = optim.lr_scheduler.ReduceLROnPlateau(optimizer_vae, 'min', patience=args.lr_scheduler_patience_vae, factor=0.1, verbose=False)

        logger.info(f"  {fold_idx_str} Entrenando VAE (Decoder: {args.decoder_type}, Encoder Layers: {args.num_conv_layers_encoder})...")
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state_dict = None
        fold_vae_train_losses, fold_vae_val_losses, fold_vae_betas = [], [], []

        for epoch in range(args.epochs_vae):
            vae_fold_k.train()
            epoch_train_loss = 0.0
            current_beta = get_cyclical_beta_schedule(epoch, args.epochs_vae, args.beta_vae, args.cyclical_beta_n_cycles, args.cyclical_beta_ratio_increase)
            
            for (data,) in vae_train_loader:
                data = data.to(device)
                optimizer_vae.zero_grad()
                recon_batch, mu, logvar, _ = vae_fold_k(data)
                loss = vae_loss_function(recon_batch, data, mu, logvar, beta=current_beta)
                loss.backward()
                optimizer_vae.step()
                epoch_train_loss += loss.item() * data.size(0)
            
            avg_epoch_train_loss = epoch_train_loss / len(vae_train_loader.dataset)
            fold_vae_train_losses.append(avg_epoch_train_loss)
            fold_vae_betas.append(current_beta)
            
            log_msg = f"  {fold_idx_str} VAE E{epoch+1}/{args.epochs_vae}, TrL: {avg_epoch_train_loss:.4f}, Beta: {current_beta:.4f}, LR: {optimizer_vae.param_groups[0]['lr']:.2e}"

            if vae_internal_val_loader:
                vae_fold_k.eval()
                epoch_val_loss = 0.0
                with torch.no_grad():
                    for (val_data,) in vae_internal_val_loader:
                        val_data = val_data.to(device)
                        recon_val, mu_val, logvar_val, _ = vae_fold_k(val_data)
                        val_loss_batch = vae_loss_function(recon_val, val_data, mu_val, logvar_val, beta=current_beta)
                        epoch_val_loss += val_loss_batch.item() * val_data.size(0)
                avg_epoch_val_loss = epoch_val_loss / len(vae_internal_val_loader.dataset)
                fold_vae_val_losses.append(avg_epoch_val_loss)
                log_msg += f", VAE_ValL: {avg_epoch_val_loss:.4f}"

                if scheduler_vae: scheduler_vae.step(avg_epoch_val_loss)
                if avg_epoch_val_loss < best_val_loss:
                    if not np.isnan(avg_epoch_val_loss):
                        best_val_loss = avg_epoch_val_loss
                        epochs_no_improve = 0
                        best_model_state_dict = copy.deepcopy(vae_fold_k.state_dict())
                else:
                    epochs_no_improve += 1

                if args.early_stopping_patience_vae > 0 and epochs_no_improve >= args.early_stopping_patience_vae:
                    val_str = f"{best_val_loss:.4f}" if not np.isnan(best_val_loss) else "N/A"
                    logger.info(f"  {fold_idx_str} Early stopping VAE en epoch {epoch+1}. Mejor val_loss: {val_str}")
                    break
            else: 
                 fold_vae_val_losses.append(np.nan) 
                 best_model_state_dict = copy.deepcopy(vae_fold_k.state_dict()) 

            if (epoch + 1) % args.log_interval_epochs_vae == 0 or epoch == args.epochs_vae - 1:
                logger.info(log_msg)
        
        if best_model_state_dict:
            vae_fold_k.load_state_dict(best_model_state_dict)
            val_string = f"{best_val_loss:.4f}" if vae_internal_val_loader and not np.isnan(best_val_loss) else "N/A - Last Epoch"
            logger.info(f"  {fold_idx_str} VAE final model loaded (best val_loss: {val_string}).")

        vae_model_fname = f"vae_model_fold_{fold_idx+1}.pt"
        torch.save(vae_fold_k.state_dict(), fold_output_dir / vae_model_fname)
        logger.info(f"  {fold_idx_str} Modelo VAE guardado en: {fold_output_dir / vae_model_fname}")
        
        if args.save_vae_training_history:
            history_data = {"train_loss": fold_vae_train_losses, "val_loss": fold_vae_val_losses, "beta": fold_vae_betas}
            joblib.dump(history_data, fold_output_dir / f"vae_train_history_fold_{fold_idx+1}.joblib")
            try:
                fig, ax1 = plt.subplots(figsize=(10,5))
                ax1.plot(history_data["train_loss"], label="Train Loss", color='blue')
                if vae_internal_val_loader and any(not np.isnan(x) for x in history_data["val_loss"]):
                    ax1.plot(history_data["val_loss"], label="VAE Val Loss", color='orange')
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Loss")
                ax1.legend(loc='upper left')
                ax2 = ax1.twinx()
                ax2.plot(history_data["beta"], label="Beta", color='green', linestyle='--')
                ax2.set_ylabel("Beta")
                ax2.legend(loc='upper right')
                plt.title(f"Fold {fold_idx+1} VAE Training History")
                plt.savefig(fold_output_dir / f"vae_train_history_fold_{fold_idx+1}.png")
                plt.close(fig)
            except Exception as e:
                logger.warning(f"  {fold_idx_str} No se pudo guardar la gráfica de historial VAE: {e}")

        all_folds_vae_history.append(history_data if args.save_vae_training_history else None)
        
        # --- Obtención de Características Latentes ---
        clf_train_dev_df = cn_ad_df.iloc[train_dev_clf_idx_in_cn_ad_df].copy()
        global_indices_clf_train_dev_all = clf_train_dev_df['tensor_idx'].values
        y_clf_train_dev_all = clf_train_dev_df['label'].values
        log_group_distributions(clf_train_dev_df, strat_cols, "Pool Train/Dev (Clasificador)", fold_idx_str)

        # --- Obtención de Características Latentes ---
        vae_fold_k.eval()
        with torch.no_grad():
            full_train_dev_tensor_norm = apply_normalization_params(
                current_global_tensor[global_indices_clf_train_dev_all],
                norm_params_fold_list
            )
            _, mu_train_dev, _, z_train_dev = vae_fold_k(
                torch.from_numpy(full_train_dev_tensor_norm).float().to(device)
            )

            X_np = mu_train_dev.cpu().numpy() if args.latent_features_type == 'mu' else z_train_dev.cpu().numpy()
            feature_names = [f"latent_{i}" for i in range(X_np.shape[1])]
            X_latent_train_dev = pd.DataFrame(X_np, columns=feature_names)

            # ---------- TEST ----------
            X_test_final_tensor_norm = apply_normalization_params(
                current_global_tensor[global_indices_clf_test_this_fold],
                norm_params_fold_list
            )
            if X_test_final_tensor_norm.shape[0] > 0:
                _, mu_test_final, _, z_test_final = vae_fold_k(
                    torch.from_numpy(X_test_final_tensor_norm).float().to(device)
                )
                X_np_test = mu_test_final.cpu().numpy() if args.latent_features_type == 'mu' else z_test_final.cpu().numpy()
                X_latent_test_final = pd.DataFrame(X_np_test, columns=feature_names)
            else:
                X_latent_test_final = pd.DataFrame(columns=feature_names)
            y_test_final = y_classifier_labels_cn_ad[test_clf_idx_in_cn_ad_df]



        # --- BUCLE DE CLASIFICADOR REFACTORIZADO ---
        for current_classifier_type in args.classifier_types:
            logger.info(f"    --- Entrenando Clasificador: {current_classifier_type} ---")
            
            try:
                # Obtenemos el pipeline completo y el grid de parámetros
                full_pipeline, param_distributions, n_iter_search = get_classifier_and_grid(
                    classifier_type=current_classifier_type,
                    seed=args.seed,
                    balance=args.classifier_use_class_weight,
                    use_smote=args.use_smote,
                    tune_sampler_params=args.tune_sampler_params,
                    mlp_hidden_layers=args.mlp_classifier_hidden_layers,
                    calibrate=args.classifier_calibrate
                )
            except (ImportError, ValueError) as e:
                logger.error(f"Error al obtener pipeline para {current_classifier_type}: {e}. Saltando.")
                continue

            # RandomizedSearchCV opera sobre todo el pipeline
            # La división interna de CV se encargará de aplicar los pasos de forma correcta
            inner_cv_for_hp_tune = StratifiedKFold(n_splits=args.inner_folds, shuffle=True, random_state=args.seed + fold_idx + 30)
            
            random_search = RandomizedSearchCV(
                estimator=full_pipeline,
                param_distributions=param_distributions,
                n_iter=n_iter_search,
                cv=inner_cv_for_hp_tune,
                scoring=args.gridsearch_scoring,
                n_jobs=args.n_jobs_gridsearch,
                verbose=0,
                random_state=args.seed + fold_idx,
                error_score='raise'
            )
            
            logger.info(f"      Ajustando HPs para {current_classifier_type} con RandomizedSearchCV (n_iter={n_iter_search})...")
            # Ajustamos usando todo el pool de train/dev del clasificador.
            # RandomizedSearchCV se encarga de la división interna y de aplicar el pipeline correctamente.
            random_search.fit(X_latent_train_dev, y_clf_train_dev_all)
            
            best_params_clf = random_search.best_params_
            logger.info(f"      Mejores HPs para {current_classifier_type}: {best_params_clf}")
            
            # El mejor estimador ya es un pipeline completo, ajustado con los mejores HPs
            # y re-entrenado en todo el set de train/dev.
            final_clf_model = random_search.best_estimator_
            
            logger.info(f"      Modelo final (pipeline) para {current_classifier_type} listo.")

            # --- EVALUACIÓN Y GUARDADO (SIMPLIFICADO) ---
            fold_results_clf = {
                'fold': fold_idx + 1, 
                'actual_classifier_type': current_classifier_type, 
                'best_clf_params': best_params_clf
            }

            if X_latent_test_final.shape[0] > 0:
                # Predecimos directamente en los datos de test latentes.
                # El pipeline `final_clf_model` se encargará de aplicar el escalado.
                y_pred_proba = final_clf_model.predict_proba(X_latent_test_final)[:, 1]
                y_pred       = final_clf_model.predict(X_latent_test_final)
                
                fold_results_clf.update({
                    'auc': roc_auc_score(y_test_final, y_pred_proba),
                    'pr_auc': average_precision_score(y_test_final, y_pred_proba),
                    'accuracy': accuracy_score(y_test_final, y_pred),
                    'balanced_accuracy': balanced_accuracy_score(y_test_final, y_pred),
                    'sensitivity': recall_score(y_test_final, y_pred, pos_label=1, zero_division=0),
                    'specificity': recall_score(y_test_final, y_pred, pos_label=0, zero_division=0),
                    'f1_score': f1_score(y_test_final, y_pred, pos_label=1, zero_division=0)
                })
                all_folds_clf_predictions.append({
                    'fold': fold_idx + 1, 'classifier_type': current_classifier_type, 
                    'y_true': y_test_final, 'y_pred_proba': y_pred_proba, 'y_pred': y_pred
                })
            else:
                metrics_to_nan = ['auc', 'pr_auc', 'accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1_score']
                for m in metrics_to_nan: fold_results_clf[m] = np.nan

            logger.info(f"      Resultados Fold {fold_idx+1} ({current_classifier_type}): AUC={fold_results_clf.get('auc',np.nan):.4f}, Bal.Acc={fold_results_clf.get('balanced_accuracy',np.nan):.4f}")
            
            if args.save_fold_artefacts:
                # Guardamos el pipeline completo
                joblib.dump(final_clf_model, fold_output_dir / f"classifier_{current_classifier_type}_pipeline_fold_{fold_idx+1}.joblib")
                logger.info(f"      Pipeline completo de {current_classifier_type} del fold {fold_idx+1} guardado.")
            
            all_folds_metrics.append(fold_results_clf)
        
        del vae_fold_k, optimizer_vae, vae_train_loader, vae_internal_val_loader, scheduler_vae, best_model_state_dict
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
        logger.info(f"  {fold_idx_str} completado en {time.time() - fold_start_time:.2f} segundos.")

    # ... (La sección final de resumen y guardado de resultados no cambia) ...
    if all_folds_metrics:
        metrics_df = pd.DataFrame(all_folds_metrics)
        
        for clf_type_iterated in args.classifier_types:
            metrics_df_clf = metrics_df[metrics_df['actual_classifier_type'] == clf_type_iterated]
            if not metrics_df_clf.empty:
                logger.info(f"\n--- Resumen de Rendimiento para Clasificador: {clf_type_iterated} (Promedio sobre Folds Externos) ---")
                for metric in ['auc', 'pr_auc', 'accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1_score']:
                    if metric in metrics_df_clf.columns and metrics_df_clf[metric].notna().any():
                        mean_val = metrics_df_clf[metric].mean()
                        std_val = metrics_df_clf[metric].std()
                        logger.info(f"{metric.capitalize():<20}: {mean_val:.4f} +/- {std_val:.4f}")
        
        main_clf_type_for_fname = args.classifier_types[0] if args.classifier_types else "genericclf"
        fname_suffix = (f"{main_clf_type_for_fname}_vae{args.decoder_type}{args.num_conv_layers_encoder}l_"
                        f"ld{args.latent_dim}_beta{args.beta_vae}_norm{args.norm_mode}_"
                        f"ch{num_input_channels_for_vae}{'sel' if args.channels_to_use else 'all'}_"
                        f"intFC{args.intermediate_fc_dim_vae}_drop{args.dropout_rate_vae}_"
                        f"ln{1 if args.use_layernorm_vae_fc else 0}_outer{args.outer_folds}x{args.repeated_outer_folds_n_repeats if args.repeated_outer_folds_n_repeats > 1 else 1}_"
                        f"score{args.gridsearch_scoring}")
        
        results_csv_path = output_base_dir / f"all_folds_metrics_MULTI_{fname_suffix}.csv"
        metrics_df.to_csv(results_csv_path, index=False)
        logger.info(f"Resultados detallados de todos los clasificadores guardados en: {results_csv_path}")

        summary_txt_path = output_base_dir / f"summary_metrics_MULTI_{fname_suffix}.txt"
        with open(summary_txt_path, 'w') as f:
            f.write(f"Run Arguments:\n{vars(args)}\n\n")
            f.write(f"Git Commit Hash: {args.git_hash}\n\n")
            for clf_type_iterated in args.classifier_types:
                metrics_df_clf = metrics_df[metrics_df['actual_classifier_type'] == clf_type_iterated]
                if not metrics_df_clf.empty:
                    f.write(f"--- Metrics Summary for Classifier: {clf_type_iterated} ---\n")
                    for metric in ['auc', 'pr_auc', 'accuracy', 'balanced_accuracy', 'sensitivity', 'specificity', 'f1_score']:
                        if metric in metrics_df_clf.columns and metrics_df_clf[metric].notna().any():
                            f.write(f"{metric.capitalize():<20}: {metrics_df_clf[metric].mean():.4f} +/- {metrics_df_clf[metric].std():.4f}\n")
                    f.write("\nFull Metrics DataFrame Description:\n")
                    f.write(metrics_df_clf.describe().to_string())
                    f.write("\n\n")
        logger.info(f"Sumario estadístico de métricas (por clasificador) guardado en: {summary_txt_path}")

        if args.save_vae_training_history and all_folds_vae_history:
             joblib.dump(all_folds_vae_history, output_base_dir / f"all_folds_vae_training_history_{fname_suffix}.joblib")
        if all_folds_clf_predictions: 
             joblib.dump(all_folds_clf_predictions, output_base_dir / f"all_folds_clf_predictions_MULTI_{fname_suffix}.joblib")

    else:
        logger.warning("No se pudieron calcular métricas para ningún fold.")

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline VAE+Clasificador para AD/CN (v1.7.0)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # ... (Los argumentos no cambian, pero se añade uno nuevo) ...
    group_data = parser.add_argument_group('Data and Paths')
    group_data.add_argument("--global_tensor_path", type=str, required=True, help="Ruta al archivo .npz del tensor global.")
    group_data.add_argument("--metadata_path", type=str, required=True, help="Ruta al archivo CSV de metadatos.")
    group_data.add_argument("--output_dir", type=str, default="./vae_clf_output_v1.7.0", help="Directorio para guardar resultados.")
    group_data.add_argument("--channels_to_use", type=str, nargs='*', default=None, help="Lista de nombres o índices de canales a usar.")

    group_cv = parser.add_argument_group('Cross-validation')
    group_cv.add_argument("--outer_folds", type=int, default=5, help="Número de folds para CV externa del clasificador.")
    group_cv.add_argument("--repeated_outer_folds_n_repeats", type=int, default=1, help="Número de repeticiones para RepeatedStratifiedKFold.")
    group_cv.add_argument("--inner_folds", type=int, default=5, help="Folds para CV interna (RandomizedSearchCV).") 
    group_cv.add_argument("--classifier_stratify_cols", type=str, nargs='*', default=['Sex'], help="Columnas adicionales para estratificación del clasificador.")
    group_cv.add_argument("--classifier_hp_tune_ratio", type=float, default=0.25, help="Proporción de datos de train/dev para ajuste de HP.")

    group_vae = parser.add_argument_group('VAE Model and Training')
    group_vae.add_argument("--num_conv_layers_encoder", type=int, default=4, choices=[3, 4], help="Capas convolucionales en encoder VAE.") 
    group_vae.add_argument("--decoder_type", type=str, default="convtranspose", choices=["upsample_conv", "convtranspose"], help="Tipo de decoder para VAE.") 
    group_vae.add_argument("--latent_dim", type=int, default=128, help="Dimensión del espacio latente VAE. (Recomendado: 128-256)")
    group_vae.add_argument("--lr_vae", type=float, default=1e-4, help="Tasa de aprendizaje VAE.")
    group_vae.add_argument("--epochs_vae", type=int, default=800, help="Épocas máximas para VAE.")
    group_vae.add_argument("--batch_size", type=int, default=32, help="Tamaño del batch.")
    group_vae.add_argument("--beta_vae", type=float, default=1.0, help="Peso KLD (beta_max para annealing).")
    group_vae.add_argument("--cyclical_beta_n_cycles", type=int, default=4, help="Ciclos para annealing de beta.")
    group_vae.add_argument("--cyclical_beta_ratio_increase", type=float, default=0.4, help="Proporción de ciclo para aumentar beta. (Recomendado: 0.4)")
    group_vae.add_argument("--weight_decay_vae", type=float, default=1e-5, help="Decaimiento de peso (L2 reg) para VAE.")
    group_vae.add_argument("--vae_final_activation", type=str, default="tanh", choices=["sigmoid", "tanh", "linear"], help="Activación final del decoder VAE.")
    group_vae.add_argument("--intermediate_fc_dim_vae", type=str, default="quarter", help="Dimensión FC intermedia en VAE ('0', 'half', 'quarter', o entero).")
    group_vae.add_argument("--dropout_rate_vae", type=float, default=0.2, help="Tasa de dropout en VAE.")
    group_vae.add_argument("--use_layernorm_vae_fc", action='store_true', help="Usar LayerNorm en capas FC del VAE.")
    group_vae.add_argument("--vae_val_split_ratio", type=float, default=0.2, help="Proporción para validación VAE.")
    group_vae.add_argument("--early_stopping_patience_vae", type=int, default=20, help="Paciencia early stopping VAE. (Recomendado: 15-20)")
    group_vae.add_argument("--lr_scheduler_patience_vae", type=int, default=15, help="Paciencia scheduler LR VAE. (Recomendado: 10-15)")

    group_clf = parser.add_argument_group('Classifier')
    
    clf_choices = get_available_classifiers()
    group_clf.add_argument(
        "--classifier_types", nargs="+", default=["rf", "svm", "gb"],
        choices=clf_choices,
        help=f"Tipos de clasificadores a entrenar. Disponibles: {', '.join(clf_choices)}"
    )
    group_clf.add_argument("--latent_features_type", type=str, default="mu", choices=["mu", "z"], help="Usar 'mu' o 'z' como features latentes.")
    group_clf.add_argument("--gridsearch_scoring", type=str, default="balanced_accuracy", help="Métrica para RandomizedSearchCV.")
    
    group_clf.add_argument("--classifier_use_class_weight", action="store_true", help="Usar class_weight='balanced' en clasificadores que lo soporten.")
    group_clf.add_argument("--classifier_calibrate", action="store_true", help="Aplicar calibración de probabilidad a los clasificadores (CalibratedClassifierCV).")
    group_clf.add_argument("--use_smote", action="store_true", help="Usar SMOTE en el pipeline. (Recomendado activar)")
    group_clf.add_argument("--tune_sampler_params", action="store_true", help="Incluir hiperparámetros de SMOTE en la búsqueda de RandomizedSearch.")
    group_clf.add_argument("--mlp_classifier_hidden_layers", type=str, default="64,16", help="Capas ocultas para el clasificador MLP.")
    
    group_general = parser.add_argument_group('General and Saving Settings')
    group_general.add_argument("--norm_mode", type=str, default="zscore_offdiag", choices=["zscore_offdiag", "minmax_offdiag"], help="Modo de normalización inter-canal.")
    group_general.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad.")
    group_general.add_argument("--num_workers", type=int, default=4, help="Workers para DataLoader.")
    group_general.add_argument("--n_jobs_gridsearch", type=int, default=4, help="Jobs para RandomizedSearchCV.")
    group_general.add_argument("--log_interval_epochs_vae", type=int, default=5, help="Intervalo de épocas para loguear VAE.")
    group_general.add_argument("--save_fold_artefacts", action='store_true', help="Guardar pipeline de clasificador de cada fold.")
    group_general.add_argument("--save_vae_training_history", action='store_true', help="Guardar historial de entrenamiento del VAE (loss, beta) por fold.")
    
    args = parser.parse_args()

    # ... (La lógica de validación de argumentos y configuración de semilla no cambia) ...
    if isinstance(args.intermediate_fc_dim_vae, str) and args.intermediate_fc_dim_vae.lower() not in ["0", "half", "quarter"]:
        try:
            args.intermediate_fc_dim_vae = int(args.intermediate_fc_dim_vae)
        except ValueError:
            logger.error(f"Valor inválido para intermediate_fc_dim_vae: {args.intermediate_fc_dim_vae}. Abortando.")
            exit(1)
    
    if not (0 <= args.vae_val_split_ratio < 1): 
        if args.vae_val_split_ratio != 0:
            logger.warning(f"vae_val_split_ratio ({args.vae_val_split_ratio}) inválido. Se usará 0.")
        args.vae_val_split_ratio = 0
    
    if args.vae_val_split_ratio == 0:
        logger.info("Sin validación VAE, early stopping y LR scheduler para VAE deshabilitados.")
        args.early_stopping_patience_vae = 0 
        args.lr_scheduler_patience_vae = 0
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        args.git_hash = git_hash
    except Exception:
        args.git_hash = "N/A"
    logger.info(f"Git commit hash: {args.git_hash}")

    logger.info("--- Configuración de la Ejecución (v1.7.0) ---")
    for arg, value in sorted(vars(args).items()):
        logger.info(f"{arg}: {value}")
    logger.info("------------------------------------")

    global_tensor_data, metadata_df_full = load_data(Path(args.global_tensor_path), Path(args.metadata_path))

    if global_tensor_data is not None and metadata_df_full is not None:
        pipeline_start_time = time.time()
        train_and_evaluate_pipeline(global_tensor_data, metadata_df_full, args)
        logger.info(f"Pipeline completo en {time.time() - pipeline_start_time:.2f} segundos.")
    else:
        logger.critical("No se pudieron cargar los datos. Abortando.")

    logger.info("--- Consideraciones Finales ---")
    logger.info(f"Normalización: '{args.norm_mode}'. Activación VAE: '{args.vae_final_activation}'. Asegurar compatibilidad.")
