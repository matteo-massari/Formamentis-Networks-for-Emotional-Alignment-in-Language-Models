# Formamentis Networks for Emotional Alignment in Language Models
This repository contains the code, data, and analysis for the paper:

**"Formamentis Networks for Emotional Alignment in Language Models"**
*Matteo Massari, April 2025*

## Abstract
This project investigates the emotional coherence of Large Language Models (LLMs) through the lens of **Forma Mentis Networks (FMNs)** — semantic-emotional graphs enriched with affective information. Using a consistent narrative prompt under both positive and negative emotional conditions, we study how temperature influences the emotional alignment of GPT-like models. Structural, emotional, and classification analyses reveal that LLMs often fail to maintain affective coherence, especially under negative prompting and high temperature values.

## Project Structure

```
├── formamentis_analysis.ipynb     # Main Jupyter notebook with all experiments
├── data/                          # Generated texts, FMN graphs, emotional scores
├── figures/                       # Confusion matrices, FMN visualizations, emotion wheels
├── paper/                         # PDF of the paper
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Methodology

* **Narrative Prompting**: A fixed monologue scenario from a 25-year-old data scientist in London, presented in both positive and negative tones.
* **Temperature Variation**: Texts are generated at temperatures 0.1, 0.7, and 1.3.
* **FMN Construction**: Outputs are parsed into Forma Mentis Networks, enriched with Plutchik's emotional categories.
* **Emotional Classification**: Texts are reclassified using absolute counts of positive/negative emotions, yielding confusion matrices for each setting.

## Key Findings

* **Structural Trends**: Higher temperature increases lexical diversity but decreases conceptual density.
* **Emotional Coherence**: Low temperature yields emotionally aligned FMNs; high temperature results in semantic drift and emotional incoherence.
* **Bias Toward Positivity**: The model struggles to express negative emotional states, showing a strong bias toward positive valence regardless of the prompt.

##Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/<your-username>/emotional-alignment-fmn.git
   cd emotional-alignment-fmn
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

## Tools & Libraries

* `emoatlas` – For emotional labeling and Plutchik’s wheel
* `networkx` – Network analysis
* `matplotlib`, `seaborn` – Visualization
* `nltk`, `spaCy` – Text preprocessing
* `pandas`, `numpy` – Data manipulation

## 📈 Example Visuals

* Plutchik Emotion Wheels (positive and negative contexts)
* FMN graphs centered on "feel" and "work"
* Confusion matrices showing classification alignment

## 📜 Citation

If you use this work, please cite:

```
@misc{massari2025formamentis,
  author = {Matteo Massari},
  title = {Formamentis Networks for Emotional Alignment in Language Models},
  year = {2025},
  url = {https://github.com/<your-username>/emotional-alignment-fmn}
}
```

## Contact

For questions or collaborations, reach out to:
📧 **[matteo.massari@gmail.com](mailto:matteo.massari62@gmail.com)**


