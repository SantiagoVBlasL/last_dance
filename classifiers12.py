#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classifiers12.py 
"""
from __future__ import annotations
import warnings
from typing import Any, Dict, List, Tuple

# Silenciar warnings de librerías
warnings.filterwarnings("ignore")
import logging
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger("catboost").setLevel(logging.ERROR)
logging.getLogger("sklearn").setLevel(logging.ERROR)
logging.getLogger("xgboost").setLevel(logging.ERROR)
from lightgbm import LGBMClassifier, basic as lgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.over_sampling import SMOTE
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution

# Configuración de GPU y cuML (solo si está disponible)
try:
    import cupy as cp
    has_gpu = cp.cuda.runtime.getDeviceCount() > 0
except (ImportError, cp.cuda.runtime.CUDARuntimeError):
    has_gpu = False

def get_available_classifiers() -> List[str]:
    """Devuelve la lista de clasificadores soportados."""
    return ["rf", "gb", "svm", "logreg", "mlp", "xgb"]

def _parse_hidden_layers(hidden_layers_str: str | None) -> Tuple[int, ...]:
    if not hidden_layers_str:
        return (128, 64)
    return tuple(int(x.strip()) for x in hidden_layers_str.split(',') if x.strip())

logger = logging.getLogger(__name__)
    
def get_classifier_and_grid(
    classifier_type: str,
    *,
    seed: int = 42,
    latent_dim: int = 512,
    metadata_cols: List[str] | None = None,
    balance: bool = False,
    use_smote: bool = False,
    tune_sampler_params: bool = False,
    mlp_hidden_layers: str = "128,64",
    calibrate: bool = False,
    calibrate_method: str = "sigmoid",   # "sigmoid"|"isotonic"
    add_feature_select: bool = False,
    fs_k_min: int = 32,
    fs_k_max: int = 256,
) -> Tuple[ImblearnPipeline, Dict[str, Any], int]:
    """
    Construye un pipeline de imblearn y devuelve el pipeline, el grid de búsqueda y el n_iter.
    """
    if metadata_cols is None:
        metadata_cols = []
    ctype = classifier_type.lower()
    if ctype not in get_available_classifiers():
        raise ValueError(f"Tipo de clasificador no soportado: {classifier_type!r}")
    
    # Nota: dejar que algunos modelos busquen class_weight en el grid suele rendir mejor cuando el desbalance es leve.
    # Aquí solo seteamos default; podemos sobre-escribir en los param_distributions de cada modelo.
    class_weight_default = "balanced" if balance else None
    model: Any
    param_distributions: Dict[str, Any] = {}
    n_iter_search = 150

    if ctype == 'svm':
        # Permitimos kernel lineal y RBF; gamma auto/scale + log-range.
        model = SVC(probability=True, random_state=seed, class_weight=class_weight_default, cache_size=500)
        param_distributions = {
            'model__C': FloatDistribution(1e-4, 1e3, log=True),
            # gamma categórico + continuo: usamos unión mediante grid condicional (Optuna maneja cond).
            'model__gamma': FloatDistribution(1e-7, 1e2, log=True),
            'model__kernel': CategoricalDistribution(['linear','rbf']),
            'model__class_weight': CategoricalDistribution([None, 'balanced']),
        }
        n_iter_search = 300

    elif ctype == 'logreg':
        # Cambiamos a solver='saga' para soportar l1, l2, elasticnet; multi_class='auto' tolera binario.
        model = LogisticRegression(
            random_state=seed,
            class_weight=class_weight_default,
            solver='saga',
            penalty='l2',   # se sobre-escribe
            max_iter=5000,
            l1_ratio=None,
        )
        param_distributions = {
            'model__C': FloatDistribution(1e-4, 1e2, log=True),
            'model__penalty': CategoricalDistribution(['l1','l2','elasticnet']),
            'model__l1_ratio': FloatDistribution(0.01, 0.95) ,  # solo usado si elasticnet
            'model__class_weight': CategoricalDistribution([None,'balanced']),
        }
        n_iter_search = 350

    elif ctype == "gb":
        model = LGBMClassifier(
            random_state=seed,
            objective="binary",
            class_weight=class_weight_default,
            n_jobs=1,
            verbose=-1,
        )

        # ---------- 1) ¿La librería fue compilada CON GPU? ----------
        gpu_available = False
        try:
            # -- si la función existe y devuelve 1, la build trae soporte
            from lightgbm.basic import _LIB, _safe_call
            if hasattr(_LIB, "LGBM_HasGPU"):
                gpu_available = bool(_safe_call(_LIB.LGBM_HasGPU()))
        except Exception:
            pass         # cualquier problema ⇒ asumimos que no

        # ---------- 2) ¿Hay hardware CUDA visible? ----------
        hw_gpu = has_gpu   # viene del bloque cupy de más arriba

        # ---------- 3) Elegir dispositivo ----------
        if gpu_available and hw_gpu:
            model.set_params(device_type="gpu", gpu_use_dp=True)
            logger.info("[LightGBM] → GPU habilitada")
        else:
            model.set_params(device_type="cpu")
            logger.info("[LightGBM] → usando CPU")

        # ---------- 4) Espacio de búsqueda ----------
        param_distributions = {
            "model__learning_rate":    FloatDistribution(0.000001, 0.2, log=True),
            "model__n_estimators":     IntDistribution(600, 3000),
            "model__class_weight":     CategoricalDistribution([None, "balanced"]),
            "model__max_depth":        IntDistribution(3, 16),
            "model__num_leaves":       IntDistribution(20, 512),
            "model__min_child_samples":IntDistribution(5, 50),
            "model__bagging_fraction": FloatDistribution(0.5, 1.0),
            "model__feature_fraction": FloatDistribution(0.5, 1.0),
            "model__bagging_freq":     IntDistribution(1, 10),
            "model__min_child_weight": FloatDistribution(1e-5, 10, log=True),
            "model__reg_alpha":        FloatDistribution(1e-8, 1.0, log=True),
            "model__reg_lambda":       FloatDistribution(1e-8, 1.0, log=True),
            'model__boosting_type': CategoricalDistribution(['gbdt','dart']),
        }
        n_iter_search = 350

    elif ctype == 'rf':
        model = RandomForestClassifier(random_state=seed, class_weight=class_weight_default, n_jobs=-1)
        param_distributions = {
            'model__n_estimators': IntDistribution(100, 1200),
            'model__max_features': CategoricalDistribution(['sqrt', 'log2', 0.2, 0.4]),
            'model__max_depth': IntDistribution(8, 50),
            'model__min_samples_split': IntDistribution(2, 30),
            'model__min_samples_leaf': IntDistribution(1, 20),
        }
        n_iter_search = 150

    elif ctype == 'mlp':
        hidden = _parse_hidden_layers(mlp_hidden_layers)
        model = MLPClassifier(random_state=seed, hidden_layer_sizes=hidden, max_iter=1000, early_stopping=True, n_iter_no_change=25)
        param_distributions = {
            'model__alpha': FloatDistribution(1e-5, 1e-1, log=True),
            'model__learning_rate_init': FloatDistribution(1e-5, 1e-2, log=True),
            'model__activation': CategoricalDistribution(['relu', 'tanh']),
            'model__solver': CategoricalDistribution(['adam', 'sgd']),
            'model__learning_rate': CategoricalDistribution(['constant', 'adaptive']),

        }
        n_iter_search = 200

    elif ctype == "xgb":
        model = XGBClassifier(random_state=seed, eval_metric="auc", n_jobs=1, verbosity=0)
        if has_gpu: model.set_params(tree_method="hist", device="cuda")
        param_distributions = {
            "model__learning_rate": FloatDistribution(0.00001, 0.3, log=True),
            "model__n_estimators": IntDistribution(600, 3000),
            "model__max_depth": IntDistribution(3, 16),
            "model__subsample": FloatDistribution(0.5, 1.0),
            "model__colsample_bytree": FloatDistribution(0.5, 1.0),
            "model__min_child_weight": FloatDistribution(1e-3, 20.0, log=True),
            "model__reg_alpha":        FloatDistribution(1e-8, 10, log=True),
            "model__reg_lambda":       FloatDistribution(1e-8, 10, log=True),
            "model__gamma": FloatDistribution(1e-8, 10, log=True),
            "model__scale_pos_weight": FloatDistribution(0.5, 5.0),
        }
        # --- Manejo de Desbalance Condicional para XGBoost ---
        # Solo buscamos scale_pos_weight si NO estamos usando SMOTE o class_weight
        if not use_smote and not balance:
            param_distributions["model__scale_pos_weight"] = FloatDistribution(0.7, 1.5)
     
        n_iter_search = 350

    # Calibración opcional.
    # Recomendaciones: NO calibrar LogReg; usar "sigmoid" con pocos datos; isotonic solo con N grande.
    if calibrate and ctype in ["svm", "gb", "rf"]:
        model = CalibratedClassifierCV(model, method=calibrate_method, cv=3)
        param_distributions = {f"model__base_estimator__{k.split('__', 1)[1]}": v for k, v in param_distributions.items()}

    # NOTA: La importación se mueve dentro de la función para mayor claridad.
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    if add_feature_select:
        # Agregamos selectK en pipeline; k se tuneará.
        # NOTA: usamos nombre 'fs' en param grid.
        # ¡Añadimos el hiperparámetro para tunear 'k'!
        param_distributions['fs__k'] = IntDistribution(fs_k_min, fs_k_max)
        fs_step = ('fs', SelectKBest(mutual_info_classif, k=min(fs_k_max, latent_dim)))
    else:
        fs_step = None

    latent_cols = [f'latent_{i}' for i in range(latent_dim)]
    numeric_meta_cols = [c for c in metadata_cols if c.lower() in ['age', 'years_of_education']]
    categorical_meta_cols = list(set(metadata_cols) - set(numeric_meta_cols))

    numeric_transformer = SklearnPipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
    categorical_transformer = SklearnPipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('latent', StandardScaler(), latent_cols),
            ('num', numeric_transformer, numeric_meta_cols),
            ('cat', categorical_transformer, categorical_meta_cols)
        ],
        remainder='drop'
    )
    
    steps_ordered = [('preprocessor', preprocessor)]
    if fs_step is not None:
        steps_ordered.append(fs_step)
    if use_smote:
        steps_ordered.append(('smote', SMOTE(random_state=seed)))
        if tune_sampler_params:
            param_distributions['smote__k_neighbors'] = IntDistribution(3, 25)
            
    steps_ordered.append(('model', model))
    
    full_pipeline = ImblearnPipeline(steps=steps_ordered)

    return full_pipeline, param_distributions, n_iter_search