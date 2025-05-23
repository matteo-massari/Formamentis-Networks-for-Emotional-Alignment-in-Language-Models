import pandas as pd
import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing
import emoatlas as et  # Assicurati che sia installato/importabile
import multiprocessing

# === Parametri globali ===
prompt_types = ["positive", "negative"]
temperature_levels = [0.1, 0.3, 0.5, 0.7, 1.0, 1.3]

# === Carica il tuo dataset ===
dataset = pd.read_csv("data/data_chat_neutral.csv", on_bad_lines='skip')

dataset["text"] = dataset["text"].str.replace(r'\bjob\b', 'work', regex=True) # I know probably is a bit like cheating, but I want to keep it simple

emos = et.EmoScores() # create a class
temperature_level = [0.1,0.7,1.3]

# === Funzione per creare le reti ===
def create_networks(prompt_type, temperature):
    filtered_df = dataset[(dataset["type of prompt"] == prompt_type) &
                          (dataset["temperature"] == temperature)].copy().reset_index(drop=True)

    fMNT_networks = []
    synonyms_networks = []
    for i, row in filtered_df.iterrows():
        fmnt = emos.formamentis_network(row["text"])
        fMNT_networks.append(fmnt)

        fmnt_synonyms = emos.formamentis_network(
            row["text"], semantic_enrichment='synonyms', multiplex=True)
        fmnt_synonyms = fmnt_synonyms.edges["synonyms"]
        synonyms_networks.append(fmnt_synonyms)

    fMNT_networks_res = emos.combine_formamentis(fMNT_networks)
    return fMNT_networks_res, synonyms_networks

# === Funzione per processare una combinazione ===
def process_condition(prompt_type, temp):
    fMNT_networks_res, _ = create_networks(prompt_type, temp)

    G = nx.Graph()
    G.add_edges_from(fMNT_networks_res.edges)

    if G.number_of_nodes() == 0:
        return {"type_of_prompt": prompt_type, "temperature": temp, "n_edges": 0, "n_nodes": 0, "density": 0}

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    return {
        "type_of_prompt": prompt_type,
        "temperature": temp,
        "n_edges": G.number_of_edges(),
        "n_nodes": G.number_of_nodes(),
        "density": nx.density(G) if G.number_of_nodes() > 1 else 0,
    }

# === MAIN ===
def main():
    print("Uso core:", multiprocessing.cpu_count())
    tasks = [(pt, t) for pt in prompt_types for t in temperature_levels]
    results = []

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(process_condition, pt, t) for pt, t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Calcolo metriche"):

            results.append(f.result())

    df = pd.DataFrame(results)
    df.to_csv("results/graph_metrics.csv", index=False)
    print("Risultati salvati in 'graph_metrics.csv'")



if __name__ == "__main__":
    main()