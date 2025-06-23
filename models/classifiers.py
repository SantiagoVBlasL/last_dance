#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classifiers.py
===============

Módulo centralizado para definir clasificadores de scikit-learn y sus grids de
búsqueda de hiperparámetros, encapsulados en un pipeline robusto con pre-procesado.

Versión: 3.1.1 - Corrección de importación
Cambios desde v3.1.0:
- Corregido `NameError` al añadir la importación de `uniform` desde `scipy.stats`.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("sklearn").setLevel(logging.ERROR)
logging.getLogger("lightgbm").setLevel(logging.ERROR)

from typing import Any, Dict, List, Tuple

# Dependencias de scikit-learn y imbalanced-learn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import loguniform, randint, uniform

# Definimos un tipo para la salida para mayor claridad
ClassifierPipelineAndGrid = Tuple[ImblearnPipeline, Dict[str, Any], int]

def get_available_classifiers() -> List[str]:
    """Devuelve la lista de clasificadores soportados."""
    return ["rf", "gb", "svm", "logreg", "mlp"]

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
    2. Un sampler opcional (SMOTE).
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
        model = SVC(probability=True, random_state=seed, class_weight=class_weight)
        param_distributions = {
            'model__C': loguniform(1e-2, 1e3),
            'model__gamma': loguniform(1e-4, 1e-1),
            'model__kernel': ['rbf'],
        }
        n_iter_search = 150
    elif ctype == 'logreg':
        model = LogisticRegression(random_state=seed, class_weight=class_weight, solver='liblinear', max_iter=2000)
        param_distributions = {'model__C': loguniform(1e-4, 1e2)}
        n_iter_search = 150
    elif ctype == 'gb':
        model = LGBMClassifier(random_state=seed, class_weight=class_weight, n_jobs=1, verbose=-1)
        param_distributions = {
            'model__num_leaves': randint(20, 150),
            'model__min_child_samples': randint(5, 30),
            'model__subsample': uniform(0.5, 0.5),
            'model__learning_rate': loguniform(0.005, 0.1),
            'model__n_estimators': randint(200, 1000),
            'model__reg_alpha': loguniform(1e-2, 10.0),
            'model__reg_lambda': loguniform(1e-2, 10.0),
        }
        n_iter_search = 150
    elif ctype == 'rf':
        model = RandomForestClassifier(random_state=seed, class_weight=class_weight, n_jobs=-1)
        param_distributions = {
            'model__n_estimators': randint(100, 800),
            'model__max_depth': [None, 10, 20, 30, 40, 50],
            'model__min_samples_split': randint(2, 20),
            'model__min_samples_leaf': randint(1, 10),
            'model__max_features': ['sqrt', 'log2', None],
        }
        n_iter_search = 200
    elif ctype == 'mlp':
        hidden = _parse_hidden_layers(mlp_hidden_layers)
        model = MLPClassifier(random_state=seed, hidden_layer_sizes=hidden, max_iter=1000, early_stopping=True, n_iter_no_change=25)
        param_distributions = {
            'model__alpha': loguniform(1e-6, 1e-1),
            'model__learning_rate_init': loguniform(1e-4, 1e-2),
        }
        n_iter_search = 200

    # --- 2. Opcionalmente, envolver el modelo en un CalibratedClassifierCV ---
    if calibrate and ctype in ['svm', 'gb', 'rf']:
        model = CalibratedClassifierCV(model, method='isotonic', cv=3)
        param_distributions = {f"model__base_estimator__{k.split('__')[1]}": v for k, v in param_distributions.items()}


    # --- 3. Construir el pipeline ---
    pipeline_steps = []
    
    if ctype in ['rf', 'gb']:
        pipeline_steps.append(('scaler', 'passthrough'))
    else:
        pipeline_steps.append(('scaler', StandardScaler()))

    if use_smote:
        pipeline_steps.append(('smote', SMOTE(random_state=seed)))
        if tune_sampler_params:
            param_distributions['smote__k_neighbors'] = randint(2, 11)
            param_distributions['smote__sampling_strategy'] = ['auto', 'minority']

    pipeline_steps.append(('model', model))

    full_pipeline = ImblearnPipeline(steps=pipeline_steps)

    return full_pipeline, param_distributions, n_iter_search