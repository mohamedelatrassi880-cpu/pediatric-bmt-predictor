import pandas as pd

def optimize_memory_v2(df):
    """
    Optimise l'utilisation de la mémoire d'un DataFrame en utilisant 
    les fonctionnalités intégrées de Pandas (pd.to_numeric).
    """
    # memory_usage(deep=True) est plus précis s'il y a des chaînes de caractères
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f'Memory usage before optimization: {start_mem:.4f} MB')

    # 1. Sélectionner et réduire les colonnes entières (int)
    int_cols = df.select_dtypes(include=['int8', 'int16', 'int32', 'int64']).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    # 2. Sélectionner et réduire les colonnes décimales (float)
    float_cols = df.select_dtypes(include=['float16', 'float32', 'float64']).columns
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast='float')

    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f'Memory usage after optimization: {end_mem:.4f} MB')
    
    # Gestion de la division par zéro au cas où le DataFrame serait vide
    if start_mem > 0:
        print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    
    return df
