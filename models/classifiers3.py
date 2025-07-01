#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classifiers3.py
===============

Módulo centralizado para definir clasificadores de scikit-learn y sus grids de
búsqueda de hiperparámetros, encapsulados en un pipeline robusto con pre-procesado.

Versión: 3.2.0 - Fallback a CPU para RandomForest
Cambios desde v3.1.1:
- Se revierte RandomForest a la implementación de scikit-learn (CPU) para
  máxima estabilidad y compatibilidad con el pipeline.
- Corregida la lógica de construcción del pipeline para evitar nombres duplicados.
- Se usa siempre SMOTE de imblearn para evitar errores de compatibilidad con GPU.
"""
from __future__ import annotations
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger("catboost").setLevel(logging.ERROR)
logging.getLogger("sklearn").setLevel(logging.ERROR)

import torch
import numpy as np 
from typing import Any, Dict, List, Tuple
import os

# 1️⃣  Configuración del logger principal
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 2️⃣  Silenciar librerías ruidosas
for noisy in ["lightgbm", "optuna", "sklearn", "xgboost"]:
    logging.getLogger(noisy).setLevel(logging.ERROR)

# Comprobación de GPU
try:
    import cupy as cp
    has_gpu = cp.cuda.runtime.getDeviceCount() > 0
except (ImportError, cp.cuda.runtime.CUDARuntimeError):
    has_gpu = False

print("¿GPU visible?:", has_gpu)

os.environ["XGB_HIDE_LOG"] = "1"

# Dependencias de scikit-learn y otras librerías
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.over_sampling import SMOTE
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution

# Definimos un tipo para la salida para mayor claridad
ClassifierPipelineAndGrid = Tuple[ImblearnPipeline, Dict[str, Any], int]

def get_available_classifiers() -> List[str]:
    """Devuelve la lista de clasificadores soportados."""
    return ["rf", "gb", "svm", "logreg", "mlp", "xgb", "cat"]

def _parse_hidden_layers(hidden_layers_str: str | None) -> Tuple[int, ...]:
    """Convierte un string '128,64' a una tupla (128, 64)."""
    if not hidden_layers_str:
        return (128, 64)
    return tuple(int(x.strip()) for x in hidden_layers_str.split(',') if x.strip())

def get_classifier_and_grid(
    classifier_type: str,
    *,
    seed: int = 42,
    balance: bool = False,
    use_smote: bool = False,
    tune_sampler_params: bool = False,
    mlp_hidden_layers: str = "128,64",
    calibrate: bool = False
) -> ClassifierPipelineAndGrid:
    """
    Construye un pipeline de imblearn y devuelve el pipeline, el grid de parámetros y el n_iter.

    El pipeline contiene:
    1. Un escalador (StandardScaler o 'passthrough').
    2. Un sampler opcional (SMOTE de imblearn).
    3. Un clasificador opcionalmente calibrado.
    """
    ctype = classifier_type.lower()
    if ctype not in get_available_classifiers():
        raise ValueError(f"Tipo de clasificador no soportado: {classifier_type!r}")

    # --- 1. Definir el modelo base y su grid de búsqueda ---
    class_weight = "balanced" if balance else None
    model: Any
    param_distributions: Dict[str, Any]
    n_iter_search = 150  # Default n_iter

    # Prefijo 'model__' para los parámetros del clasificador dentro del pipeline
    if ctype == 'svm':
        model = SVC(probability=True, random_state=seed, class_weight=class_weight, cache_size=500) # cache_size puede ayudar
        param_distributions = {
            # Rango más amplio para C, permitiendo regularización más fuerte o más débil
            'model__C': FloatDistribution(1e-2, 1e4, log=True), 
            # Rango más amplio para gamma, a menudo el valor óptimo es muy pequeño
            'model__gamma': FloatDistribution(1e-7, 1e-1, log=True),
            'model__kernel': CategoricalDistribution(['rbf']),
        }
        n_iter_search = 160

    elif ctype == 'logreg':
        model = LogisticRegression(random_state=seed, class_weight=class_weight, solver='liblinear', max_iter=2000)
        param_distributions = {
            # El rango actual es bueno, no necesita cambios drásticos
            'model__C': FloatDistribution(1e-5, 1e3, log=True)
        }
        n_iter_search = 140

    elif ctype == 'gb':
        model = LGBMClassifier(random_state=seed, class_weight=class_weight, n_jobs=1, verbose=-1)
        if has_gpu:
            try:
                # Comprobación limpia: preguntar a la librería si trae soporte GPU
                from lightgbm.basic import _LIB, _safe_call
                if bool(_safe_call(_LIB.LGBM_HasGPU())):
                    model.set_params(device='gpu', gpu_use_dp=True)
                    print("[LightGBM] ➜  Se usará GPU")
                else:
                    print("[LightGBM] ⚠  Build sin soporte GPU, usando CPU")
            except Exception:
                print("[LightGBM] ⚠  No se pudo verificar el soporte de GPU, usando CPU")
        
        param_distributions = {
            'model__num_leaves': IntDistribution(20, 150), # Un límite inferior más bajo
            'model__min_child_samples': IntDistribution(5, 40), # Un límite inferior más bajo
            'model__subsample': FloatDistribution(0.4, 1.0),
            'model__learning_rate': FloatDistribution(5e-4, 0.1, log=True), # Explora tasas más bajas
            'model__n_estimators': IntDistribution(300, 2000), # Más estimadores
            'model__reg_alpha': FloatDistribution(1e-3, 50.0, log=True), # Rango más amplio
            'model__reg_lambda': FloatDistribution(1e-3, 50.0, log=True), # Rango más amplio
        }
        n_iter_search = 120

    elif ctype == 'rf':
        print("[RandomForest] ➜ Usando implementación de scikit-learn (CPU).")
        model = RandomForestClassifier(random_state=seed, class_weight=class_weight, n_jobs=-1)
        param_distributions = {
            'model__n_estimators': IntDistribution(100, 1000),
            'model__max_features': CategoricalDistribution(['sqrt', 'log2', 0.2, 0.4]), # Añadir proporciones numéricas
            # Explorar más granularmente la profundidad
            'model__max_depth': IntDistribution(8, 50),
            'model__min_samples_split': IntDistribution(2, 30), 
            'model__min_samples_leaf': IntDistribution(1, 20)
        }
        n_iter_search = 100
    
    elif ctype == 'mlp':
        hidden = _parse_hidden_layers(mlp_hidden_layers)
        model = MLPClassifier(random_state=seed, hidden_layer_sizes=hidden, max_iter=1000, early_stopping=True, n_iter_no_change=25)
        param_distributions = {
            # Ampliar un poco el rango de regularización
            'model__alpha': FloatDistribution(1e-7, 1e-2, log=True),
            'model__learning_rate_init': FloatDistribution(1e-5, 1e-2, log=True),
        }
        n_iter_search = 120

    elif ctype == "xgb":
        # Sugerido: explícitamente gpu_hist y gpu_id
        model = XGBClassifier(
            random_state=seed,
            eval_metric="auc",
            n_jobs=1,
            tree_method="hist",        # GPU/CPU se decide con 'device'
            device="cuda",           # GPU
            verbosity=0
        )
        if has_gpu:
            print("[XGBoost] ➜  Se usará GPU (device=cuda)")
        else:
            print("[XGBoost] ⚠  GPU no disponible, usando CPU.")

        param_distributions = {
            "model__n_estimators": IntDistribution(150, 1600), # Rango ajustado
            "model__learning_rate": FloatDistribution(1e-5, 0.1, log=True),
            "model__max_depth": IntDistribution(3, 12),
            "model__subsample": FloatDistribution(0.3, 1.1),
            "model__colsample_bytree": FloatDistribution(0.1, 1.2),
            # min_child_weight es un parámetro de regularización importante
            "model__min_child_weight": FloatDistribution(0.5, 20, log=True),
        }
        n_iter_search = 200

    elif ctype == "cat":
        model = CatBoostClassifier(random_state=seed, eval_metric="Logloss", verbose=0, loss_function="Logloss", thread_count=1)
        if has_gpu:
            model.set_params(task_type="GPU", devices="0:0")
            print("[CatBoost] ➜  Se usará GPU")
        else:
            print("[CatBoost] ⚠  GPU no disponible, usando CPU.")

        param_distributions = {
            "model__depth": IntDistribution(4, 8),
            "model__learning_rate": FloatDistribution(1e-3, 0.08, log=True),
            "model__l2_leaf_reg": FloatDistribution(0.1, 20.0, log=True),
            "model__iterations": IntDistribution(400, 1500),
            "model__bagging_temperature": FloatDistribution(0.1, 0.9),
        }
        n_iter_search = 120


    # --- 2. Opcionalmente, envolver el modelo en un CalibratedClassifierCV ---
    if calibrate and ctype in ["svm", "gb", "rf"]:
        # ... (la lógica de calibración no necesita cambios) ...
        model = CalibratedClassifierCV(model, method="isotonic", cv=3)
        _cal = CalibratedClassifierCV(model.model if hasattr(model, "model") else model)
        _inner = "estimator" if "estimator" in _cal.get_params() else "base_estimator"
        param_distributions = { f"model__{_inner}__{k.split('__', 1)[1]}": v for k, v in param_distributions.items() }

    # --- 3. Construir el pipeline (LÓGICA CORREGIDA Y SIMPLIFICADA) ---
    pipeline_steps = []

    # Paso 1: Escalador
    # Los modelos basados en árboles no requieren escalado de características
    if ctype in ['rf', 'gb', 'xgb', 'cat']:
        pipeline_steps.append(('scaler', 'passthrough'))
    else:
        pipeline_steps.append(('scaler', StandardScaler()))

    # Paso 2: SMOTE (opcional, siempre desde imblearn para máxima compatibilidad)
    if use_smote:
        logger.info(f"[SMOTE] ➜ Usando implementación de imblearn (CPU) para el clasificador '{ctype}'.")
        pipeline_steps.append(('smote', SMOTE(random_state=seed)))# Usar n_jobs para acelerar

        if tune_sampler_params:
            # k_neighbors debe ser menor que el número de muestras en la clase minoritaria.
            # Un rango de 3 a 25 es generalmente seguro y efectivo para tamaños de dataset moderados.
            param_distributions['smote__k_neighbors'] = IntDistribution(3, 25)
    

    # Paso 3: Modelo
    pipeline_steps.append(('model', model))
    full_pipeline = ImblearnPipeline(steps=pipeline_steps)
    return full_pipeline, param_distributions, n_iter_search