from __future__ import annotations

import random
import pandas as pd

from .datasets import collect_image_records

# --- FUNCIÓN DE MUESTREO EQUILIBRADO ---

def select_balanced_sample(
    dataset_key: str,
    *,
    per_group: int = 4,                       # Número de fotos a elegir por cada tipo de postura
    splits: tuple[str, ...] = ("train", "valid", "test"), # Fases de datos permitidas
    seed: int = 7,                            # Semilla para que el sorteo sea siempre igual
) -> pd.DataFrame:
    """
    Crea una muestra donde cada grupo (postura) tiene exactamente la misma cantidad de imágenes.
    Evita que el análisis se sesgue por tener demasiadas fotos de un solo tipo.
    """
    # 1. Obtenemos todos los registros del catálogo de datasets
    records = collect_image_records(dataset_key)
    if not records:
        return pd.DataFrame(columns=["image_path", "group", "split"])

    # 2. Inicializamos el generador aleatorio con la semilla fijada
    rng = random.Random(seed)
    
    # 3. Filtramos las imágenes que pertenecen a los splits deseados
    filtered = [record for record in records if record["split"] in splits]
    
    # 4. Agrupamos las imágenes por su etiqueta de postura (group)
    grouped: dict[str, list[dict]] = {}
    for record in filtered:
        grouped.setdefault(str(record["group"]), []).append(record)

    # 5. Selección aleatoria por grupo
    selected = []
    for group_name in sorted(grouped):
        group_records = grouped[group_name][:] # Copiamos la lista para no alterar la original
        rng.shuffle(group_records)             # Mezclamos aleatoriamente
        selected.extend(group_records[:per_group]) # Tomamos solo las N primeras imágenes

    # 6. Ordenamos el resultado final para que sea predecible
    selected.sort(key=lambda item: (str(item["group"]), str(item["split"]), str(item["image_path"])))
    
    sample_df = pd.DataFrame(selected)
    if sample_df.empty:
        return pd.DataFrame(columns=["image_path", "group", "split"])
    return sample_df.reset_index(drop=True)


# --- SELECTOR GENERAL DE REGISTROS ---

def select_execution_records(
    dataset_key: str,
    *,
    strategy: str = "full_dataset",           # 'full_dataset' para todo, 'balanced' para muestra
    per_group: int = 4,
    max_images: int | None = None,            # Límite máximo total de imágenes
    splits: tuple[str, ...] = ("train", "valid", "test"),
    seed: int = 7,
) -> pd.DataFrame:
    """
    Función de alto nivel para decidir qué imágenes entrarán en el pipeline de ejecución
    Permite elegir entre procesar el dataset completo o una selección reducida
    """
    records = collect_image_records(dataset_key)
    if not records:
        return pd.DataFrame(columns=["image_path", "group", "split"])

    # Filtro inicial por split
    filtered = [record for record in records if record["split"] in splits]
    
    # Aplicamos la estrategia elegida
    if strategy == "balanced":
        sample_df = select_balanced_sample(
            dataset_key,
            per_group=per_group,
            splits=splits,
            seed=seed,
        )
    elif strategy == "full_dataset":
        sample_df = pd.DataFrame(filtered)
    else:
        raise ValueError(f"Estrategia desconocida: {strategy}")

    if sample_df.empty:
        return pd.DataFrame(columns=["image_path", "group", "split"])

    # --- LÓGICA DE RECORTE (MAX_IMAGES) ---
    # Si la muestra es más grande que el límite máximo, recortamos de forma inteligente
    if max_images is not None and max_images < len(sample_df):
        rng = random.Random(seed)
        records = sample_df.to_dict(orient="records")
        
        if strategy == "balanced":
            # Si queremos mantener el equilibrio al recortar, usamos un sistema de 'Round Robin'
            grouped: dict[tuple[str, str], list[dict]] = {}
            for record in records:
                grouped.setdefault((str(record["split"]), str(record["group"])), []).append(record)
            
            for items in grouped.values():
                rng.shuffle(items)
            
            selected: list[dict] = []
            ordered_keys = sorted(grouped)
            # Vamos sacando una imagen de cada grupo alternativamente hasta llegar al tope
            while len(selected) < max_images:
                progress_made = False
                for key in ordered_keys:
                    bucket = grouped[key]
                    if not bucket:
                        continue
                    selected.append(bucket.pop())
                    progress_made = True
                    if len(selected) >= max_images:
                        break
                if not progress_made:
                    break
            sample_df = pd.DataFrame(selected)
        else:
            # Si no es balanceado, simplemente mezclamos y cortamos
            rng.shuffle(records)
            sample_df = pd.DataFrame(records[:max_images])

    # Devolvemos el DataFrame final ordenado para facilitar la lectura del reporte
    return sample_df.sort_values(["group", "split", "image_path"]).reset_index(drop=True)