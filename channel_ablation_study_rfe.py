# -*- coding: utf-8 -*-
"""
channel_ablation_study_rfe.py
Versión: 2.2 - RFE con estado (Stateful) para resiliencia y manejo de memoria.
Cambios:
- La función RFE ahora guarda el progreso en un CSV después de cada iteración.
- Si el script se detiene, puede ser reanudado y continuará desde el último punto de control.
- Se optimizó el paso de argumentos para asegurar la consistencia de los nombres de canales.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
import gc
import copy
import subprocess
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ks_2samp, entropy
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from serentipia3 import load_data, train_and_evaluate_pipeline

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- Las funciones de análisis estadístico y univariado no cambian ---
def analyze_statistical_divergence(
    tensor: np.ndarray,
    df: pd.DataFrame,
    channel_names: list[str]
) -> pd.DataFrame:
    logger.info("--- Iniciando Análisis de Divergencia Estadística (KL y KS) ---")
    # (El código de esta función no necesita cambios)
    results = []
    ad_indices = df[df['ResearchGroup_Mapped'] == 'AD']['tensor_idx'].values
    cn_indices = df[df['ResearchGroup_Mapped'] == 'CN']['tensor_idx'].values
    off_diag_mask = ~np.eye(tensor.shape[2], dtype=bool)
    for i, channel_name in enumerate(tqdm(channel_names, desc="Analizando Divergencia de Canales")):
        ad_vals = tensor[ad_indices, i, :, :][:, off_diag_mask].flatten()
        cn_vals = tensor[cn_indices, i, :, :][:, off_diag_mask].flatten()
        ks_stat, _ = ks_2samp(ad_vals, cn_vals)
        min_val, max_val = min(ad_vals.min(), cn_vals.min()), max(ad_vals.max(), cn_vals.max())
        bins = np.linspace(min_val, max_val, num=101)
        p, _ = np.histogram(ad_vals, bins=bins, density=True)
        q, _ = np.histogram(cn_vals, bins=bins, density=True)
        p, q = p + 1e-10, q + 1e-10
        kl_divergence_symmetric = (entropy(p, q) + entropy(q, p)) / 2.0
        results.append({'channel': channel_name, 'ks_statistic': ks_stat, 'kl_divergence_sym': kl_divergence_symmetric})
    return pd.DataFrame(results)

def analyze_univariate_performance(
    tensor: np.ndarray,
    df: pd.DataFrame,
    channel_names: list[str]
) -> pd.DataFrame:
    logger.info("--- Iniciando Análisis de Rendimiento Predictivo Univariado (AUC) ---")
    # (El código de esta función no necesita cambios)
    results = []
    y = df[df['ResearchGroup_Mapped'].isin(['AD', 'CN'])]['label'].values
    subject_indices = df[df['ResearchGroup_mailed'].isin(['AD', 'CN'])]['tensor_idx'].values
    for i, channel_name in enumerate(tqdm(channel_names, desc="Analizando Rendimiento Univariado")):
        X_channel = tensor[subject_indices, i, :, :].reshape(len(subject_indices), -1)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_aucs = []
        for train_idx, test_idx in cv.split(X_channel, y):
            X_train, X_test = X_channel[train_idx], X_channel[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            scaler = StandardScaler()
            X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
            model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', max_iter=1000)
            model.fit(X_train_scaled, y_train)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            fold_aucs.append(roc_auc_score(y_test, y_pred_proba))
        results.append({'channel': channel_name, 'univariate_auc_mean': np.mean(fold_aucs)})
    return pd.DataFrame(results)

def analyze_model_ablation(
    global_tensor: np.ndarray,
    metadata_df: pd.DataFrame,
    channel_names: list[str],
    args: argparse.Namespace
) -> tuple[pd.DataFrame, float]:
    logger.info(f"--- Iniciando Estudio de Ablación para {len(channel_names)} canales ---")
    results = []
    
    baseline_args = copy.deepcopy(args)
    # MODIFICADO: Mapeo consistente usando la lista maestra
    channels_to_use_indices = [args.all_original_channel_names.index(name) for name in channel_names]
    baseline_args.channels_to_use = channels_to_use_indices
    baseline_args.output_dir = str(Path(args.output_dir) / f"rfe_step_{len(channel_names)}_channels")
    
    baseline_metrics_df = train_and_evaluate_pipeline(global_tensor, metadata_df, baseline_args)
    if baseline_metrics_df is None or baseline_metrics_df.empty:
        logger.error("La ejecución de referencia (baseline) falló. No se puede continuar con la ablación.")
        return pd.DataFrame(), 0.0
        
    baseline_performance = baseline_metrics_df[args.gridsearch_scoring].mean()
    logger.info(f"Rendimiento de referencia ({len(channel_names)} canales): {baseline_performance:.4f}")
    
    for channel_to_ablate in tqdm(channel_names, desc=f"Ablación de {len(channel_names)} canales"):
        temp_channels_to_use_names = [name for name in channel_names if name != channel_to_ablate]
        temp_channels_to_use_indices = [args.all_original_channel_names.index(name) for name in temp_channels_to_use_names]

        ablation_args = copy.deepcopy(args)
        ablation_args.channels_to_use = temp_channels_to_use_indices
        ablation_args.output_dir = str(Path(baseline_args.output_dir) / f"ablation_{channel_to_ablate.replace(' ', '_')}")

        ablation_metrics_df = train_and_evaluate_pipeline(global_tensor, metadata_df, ablation_args)
        
        if ablation_metrics_df is None or ablation_metrics_df.empty:
            logger.warning(f"La ejecución de ablación para '{channel_to_ablate}' falló. Se registrará un rendimiento de 0.")
            ablated_performance = 0.0
        else:
            ablated_performance = ablation_metrics_df[args.gridsearch_scoring].mean()

        performance_drop = baseline_performance - ablated_performance
        results.append({'channel': channel_to_ablate, 'performance_drop': performance_drop, 'ablated_performance': ablated_performance})
        gc.collect()

    return pd.DataFrame(results), baseline_performance

def run_recursive_feature_elimination(global_tensor, metadata_df, initial_channel_names, args):
    """
    ## NUEVO: RFE con estado para guardar y reanudar el progreso.
    """
    logger.info("--- INICIANDO ESTUDIO DE ELIMINACIÓN RECURSIVA DE CARACTERÍSTICAS (RFE) ---")
    
    summary_path = Path(args.output_dir) / "rfe_summary.csv"
    rfe_summary_df = pd.DataFrame()
    current_channel_names = list(initial_channel_names)

    # Cargar progreso si existe
    if summary_path.exists():
        logger.warning(f"Se encontró un archivo de resumen previo en {summary_path}. Reanudando progreso.")
        rfe_summary_df = pd.read_csv(summary_path)
        if not rfe_summary_df.empty:
            last_run_channels_str = rfe_summary_df.sort_values(by='num_channels', ascending=True).iloc[0]['channels_in_baseline']
            current_channel_names = [name.strip() for name in last_run_channels_str.split(',')]
            logger.info(f"Reanudando RFE con {len(current_channel_names)} canales.")
    
    while len(current_channel_names) > 1:
        # Verificar si este paso ya se completó
        if not rfe_summary_df.empty and len(current_channel_names) in rfe_summary_df['num_channels'].values:
            logger.info(f"El paso con {len(current_channel_names)} canales ya existe en el resumen. Saltando.")
            # Eliminar el peor canal de la ejecución previa para continuar
            channel_to_remove = "ELIMINAR_PLACEHOLDER" # Lógica para determinar esto si es necesario
            # Esta parte es compleja, por ahora, lo más simple es requerir que el usuario borre el último paso si quiere re-ejecutarlo.
            # Aquí simplemente avanzamos al siguiente estado.
            last_baseline_df = rfe_summary_df[rfe_summary_df['num_channels'] == len(current_channel_names)]
            channels_from_last_run = set(last_baseline_df['channels_in_baseline'].iloc[0].split(', '))

            next_step_df = rfe_summary_df[rfe_summary_df['num_channels'] == len(current_channel_names) - 1]
            if next_step_df.empty:
                logger.error("No se puede determinar el siguiente paso. Por favor, revise el CSV de resumen.")
                break

            channels_from_next_run = set(next_step_df['channels_in_baseline'].iloc[0].split(', '))
            removed_channel_set = channels_from_last_run - channels_from_next_run
            
            if not removed_channel_set:
                 logger.error(f"No se pudo determinar qué canal fue eliminado para pasar de {len(current_channel_names)} a {len(current_channel_names)-1} canales. Deteniendo.")
                 break
            
            channel_to_remove = list(removed_channel_set)[0]
            logger.info(f"Canal eliminado en la ejecución previa: '{channel_to_remove}'")
            current_channel_names.remove(channel_to_remove)
            continue


        logger.info(f"--- Iniciando iteración RFE con {len(current_channel_names)} canales ---")
        logger.info(f"Canales actuales: {current_channel_names}")

        ablation_results_df, current_baseline_perf = analyze_model_ablation(
            global_tensor, metadata_df, current_channel_names, args
        )

        if ablation_results_df.empty:
            logger.error("La iteración de RFE falló porque la ablación no devolvió resultados. Deteniendo.")
            break
        
        # Guardar el resultado de la línea base actual
        new_row = {
            'num_channels': len(current_channel_names),
            'balanced_accuracy': current_baseline_perf,
            'channels_in_baseline': ", ".join(sorted(current_channel_names))
        }
        rfe_summary_df = pd.concat([rfe_summary_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # ## PUNTO DE CONTROL: Guardar el progreso ##
        rfe_summary_df.to_csv(summary_path, index=False)
        logger.info(f"Progreso de RFE guardado en: {summary_path}")

        # Determinar el peor canal (el que menos impacta negativamente al ser removido)
        channel_to_remove_row = ablation_results_df.loc[ablation_results_df['performance_drop'].idxmin()]
        channel_to_remove = channel_to_remove_row['channel']
        
        logger.info(f"El mejor candidato para eliminar es '{channel_to_remove}', "
                    f"ya que su eliminación cambia el rendimiento en {-(channel_to_remove_row['performance_drop']):.4f}")
        
        current_channel_names.remove(channel_to_remove)
        gc.collect() # Forzar limpieza de memoria

    # Ejecución final con el último canal restante
    if len(current_channel_names) == 1 and 1 not in rfe_summary_df['num_channels'].values:
        logger.info(f"--- Evaluando el último canal restante: {current_channel_names[0]} ---")
        final_args = copy.deepcopy(args)
        final_args.channels_to_use = [args.all_original_channel_names.index(current_channel_names[0])]
        final_args.output_dir = str(Path(args.output_dir) / "rfe_step_1_channel")
        
        final_metrics_df = train_and_evaluate_pipeline(global_tensor, metadata_df, final_args)

        if final_metrics_df is not None and not final_metrics_df.empty:
            final_performance = final_metrics_df[args.gridsearch_scoring].mean()
            new_row = {
                'num_channels': 1,
                'balanced_accuracy': final_performance,
                'channels_in_baseline': current_channel_names[0]
            }
            rfe_summary_df = pd.concat([rfe_summary_df, pd.DataFrame([new_row])], ignore_index=True)
            rfe_summary_df.to_csv(summary_path, index=False) # Guardado final
            logger.info(f"Progreso final de RFE guardado en: {summary_path}")


    return rfe_summary_df.sort_values(by='num_channels', ascending=False)


def main():
    parser = argparse.ArgumentParser(
        description="Estudio de Ablación y RFE de Canales (v2.2 - con estado).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ... (El resto del parser no necesita cambios, ya que está bien estructurado)
    # Solo asegúrate de que todos los argumentos de serentipia3.py estén aquí.
    # Por brevedad, se omite aquí, pero tu script original ya lo tiene correcto.
    group_ablation = parser.add_argument_group('Ablation and RFE Study Settings')
    group_ablation.add_argument("--global_tensor_path", type=str, required=True)
    group_ablation.add_argument("--metadata_path", type=str, required=True)
    group_ablation.add_argument("--output_dir", type=str, default="./channel_ablation_results")
    
    study_type_group = group_ablation.add_mutually_exclusive_group(required=True)
    study_type_group.add_argument("--run-full-ablation", action='store_true', help="Ejecutar el estudio de ablación simple (leave-one-out).")
    study_type_group.add_argument("--run-rfe", action='store_true', help="Ejecutar el estudio de eliminación recursiva de características (RFE).")

    # El resto de los argumentos de serentipia no cambian
    group_serentipia = parser.add_argument_group('Serentipia Pipeline Arguments')
    group_serentipia.add_argument("--outer_folds", type=int, default=3)
    group_serentipia.add_argument("--repeated_outer_folds_n_repeats", type=int, default=1)
    group_serentipia.add_argument("--inner_folds", type=int, default=3)
    group_serentipia.add_argument("--classifier_stratify_cols", type=str, nargs='*', default=['Sex'])
    group_serentipia.add_argument("--num_conv_layers_encoder", type=int, default=4, choices=[3, 4])
    group_serentipia.add_argument("--decoder_type", type=str, default="convtranspose", choices=["upsample_conv", "convtranspose"])
    group_serentipia.add_argument("--latent_dim", type=int, default=512)
    group_serentipia.add_argument("--lr_vae", type=float, default=1e-4)
    group_serentipia.add_argument("--epochs_vae", type=int, default=550)
    group_serentipia.add_argument("--batch_size", type=int, default=64)
    group_serentipia.add_argument("--beta_vae", type=float, default=2.5)
    group_serentipia.add_argument("--cyclical_beta_n_cycles", type=int, default=4)
    group_serentipia.add_argument("--cyclical_beta_ratio_increase", type=float, default=0.4)
    group_serentipia.add_argument("--weight_decay_vae", type=float, default=1e-5)
    group_serentipia.add_argument("--vae_final_activation", type=str, default="tanh")
    group_serentipia.add_argument("--intermediate_fc_dim_vae", type=str, default="quarter")
    group_serentipia.add_argument("--dropout_rate_vae", type=float, default=0.2)
    group_serentipia.add_argument("--use_layernorm_vae_fc", action='store_true')
    group_serentipia.add_argument("--vae_val_split_ratio", type=float, default=0.2)
    group_serentipia.add_argument("--early_stopping_patience_vae", type=int, default=20)
    group_serentipia.add_argument("--lr_scheduler_type", type=str, default="plateau", choices=["plateau", "cosine_warm"], help="Tipo de scheduler para el VAE.")
    group_serentipia.add_argument("--lr_scheduler_T0", type=int, default=50, help="Épocas para el primer reinicio en CosineAnnealingWarmRestarts.")
    group_serentipia.add_argument("--lr_scheduler_eta_min", type=float, default=1e-7, help="Tasa de aprendizaje mínima para CosineAnnealingWarmRestarts.")
    group_serentipia.add_argument("--classifier_types", nargs="+", default=["xgb"])
    group_serentipia.add_argument("--latent_features_type", type=str, default="mu")
    group_serentipia.add_argument("--gridsearch_scoring", type=str, default="balanced_accuracy")
    group_serentipia.add_argument("--classifier_use_class_weight", action='store_true')
    group_serentipia.add_argument("--classifier_calibrate", action='store_true')
    group_serentipia.add_argument("--use_smote", action='store_true')
    group_serentipia.add_argument("--tune_sampler_params", action='store_true')
    group_serentipia.add_argument("--use_optuna_pruner", action='store_true', help="Usar Optuna's MedianPruner para HPO.")
    group_serentipia.add_argument("--mlp_classifier_hidden_layers", type=str, default="128,32")
    group_serentipia.add_argument("--metadata_features", nargs="*", default=None)
    group_serentipia.add_argument("--norm_mode", type=str, default="zscore_offdiag")
    group_serentipia.add_argument("--seed", type=int, default=42)
    group_serentipia.add_argument("--num_workers", type=int, default=4)
    group_serentipia.add_argument("--n_jobs_gridsearch", type=int, default=-1)
    group_serentipia.add_argument("--log_interval_epochs_vae", type=int, default=50)
    group_serentipia.add_argument("--save_fold_artefacts", action='store_false')
    group_serentipia.add_argument("--save_vae_training_history", action='store_false')
    group_serentipia.add_argument("--channels_to_use", nargs="*", default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # (El resto del main no necesita cambios)
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        args.git_hash = git_hash
    except Exception:
        args.git_hash = "N/A"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    global_tensor, metadata_df = load_data(Path(args.global_tensor_path), Path(args.metadata_path))
    if global_tensor is None: return
    
    with np.load(args.global_tensor_path, allow_pickle=True) as data:
        try:
            channel_names = data['channel_names'].tolist()
            args.all_original_channel_names = channel_names
        except KeyError:
            logger.warning("No se encontró 'channel_names' en el tensor. Usando nombres genéricos.")
            channel_names = [f"RawChan{i}" for i in range(global_tensor.shape[1])]
            args.all_original_channel_names = channel_names

    logger.info(f"Canales encontrados en el tensor: {channel_names}")
    ad_cn_df = metadata_df[metadata_df['ResearchGroup_Mapped'].isin(['AD', 'CN'])].copy()
    ad_cn_df['label'] = ad_cn_df['ResearchGroup_Mapped'].map({'CN': 0, 'AD': 1})

    if args.run_rfe:
        rfe_results_df = run_recursive_feature_elimination(global_tensor, metadata_df, channel_names, args)
        if not rfe_results_df.empty:
            logger.info("\n\n--- RESULTADOS FINALES DEL ESTUDIO RFE ---")
            print(rfe_results_df.to_string())
            results_path = output_dir / "rfe_summary.csv" # Guardar con el mismo nombre que usa la función
            rfe_results_df.to_csv(results_path, index=False)
            logger.info(f"\nResultados de RFE guardados en: {results_path}")

            plot_path = output_dir / "rfe_channel_performance.png"
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=rfe_results_df, x='num_channels', y='balanced_accuracy', marker='o')
            plt.title('Rendimiento del Modelo vs. Número de Canales (RFE)', fontsize=16)
            plt.xlabel('Número de Canales en el Modelo', fontsize=12)
            plt.ylabel(f"{args.gridsearch_scoring} Promedio", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.gca().invert_xaxis()
            plt.savefig(plot_path)
            logger.info(f"Gráfica de RFE guardada en: {plot_path}")
            
    elif args.run_full_ablation:
        # (La lógica para --run-full-ablation no necesita cambios)
        pass

if __name__ == "__main__":
    main()