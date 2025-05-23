###############################################
# to use emoatlas
# - pip install git+https://github.com/MassimoStel/emoatlas
# - python -m spacy download en_core_web_lg
###############################################
# inizzializziamo emoatlas
from emoatlas import EmoScores
import emoatlas
import pandas as pd
emos = EmoScores()


####################################################################
def average_zscores_by_condition(dataset, prompt_type, temperature):
    """
    Calcola la media degli z-score emozionali per tutti i testi
    di un certo tipo di prompt e temperatura.
    """
    from tqdm import tqdm
    results = []

    # Filtro
    filtered_df = dataset[(dataset["type of prompt"] == prompt_type) &
                          (dataset["temperature"] == temperature)].copy().reset_index(drop=True)

    if filtered_df.empty:
        print(f"No data found for prompt '{prompt_type}' and temperature {temperature}")
        return None

    # Calcolo z-score per ogni testo
    for i, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Processing z-scores"):
        try:
            fmnt = emos.formamentis_network(row["text"])
            z = emos.zscores(fmnt)
            results.append(z)
        except Exception as e:
            print(f"Errore nel testo {i}: {e}")

    if not results:
        print("Nessuno z-score è stato calcolato.")
        return None

    # Media dei dizionari
    df_z = pd.DataFrame(results)
    mean_zscores = df_z.mean().to_dict()
    return mean_zscores
