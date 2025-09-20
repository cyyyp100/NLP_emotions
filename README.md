# NLP Emotions

Projet de classification d'émotions à partir de tweets en anglais. Le dépôt fournit
un ensemble de notebooks exploratoires ainsi qu'un script reproductible pour entraîner
un réseau de neurones utilisant une représentation « Bag-of-Words ».

## Structure du dépôt

```
.
├── data/           # Jeux de données (échantillons Kaggle)
├── docs/           # Documentation supplémentaire et rapport de projet
├── models/         # Modèles Keras exportés et vectoriseurs
├── notebooks/      # Notebooks Jupyter d'exploration et de prototypage
└── src/            # Scripts Python réutilisables
```

Les fichiers générés par l'entraînement (modèle, vectoriseur, métriques) sont conservés
dans `models/` afin de faciliter leur réutilisation.

## Prérequis

- Python 3.9 ou ultérieur
- Environnement virtuel recommandé (`venv`, `conda`, `poetry`, ...)

Installez les dépendances nécessaires :

```bash
python -m venv .venv
source .venv/bin/activate  # sous Windows : .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Jeux de données

Un extrait du jeu de données Kaggle est disponible dans `data/text.csv`. Il contient
un sous-ensemble de tweets et leurs étiquettes numériques associées. Chaque étiquette
correspond à l'une des émotions suivantes :

| Label | Emotion   |
|-------|-----------|
| 0     | Tristesse |
| 1     | Joie      |
| 2     | Amour     |
| 3     | Colère    |
| 4     | Peur      |
| 5     | Surprise  |

Pour travailler sur le jeu de données complet, remplacez `data/text.csv` par la version
intégrale téléchargée sur Kaggle en conservant la même structure de colonnes.

## Entraîner le modèle Bag-of-Words

Le script `src/train_bow_ann.py` encapsule le pipeline d'entraînement. Exemple
d'utilisation :

```bash
python src/train_bow_ann.py \
  --data data/text.csv \
  --sample-size 5000 \
  --epochs 5 \
  --batch-size 32 \
  --predict "I feel incredibly happy today!"
```

Arguments principaux :

- `--sample-size` limite le nombre de lignes utilisées (mettre `-1` pour désactiver) ;
- `--epochs` et `--batch-size` contrôlent l'entraînement TensorFlow ;
- `--predict` affiche une prédiction d'exemple après l'entraînement ;
- `--model-path` et `--vectorizer-path` définissent les chemins de sauvegarde.

Le script produit un modèle Keras (`.keras`) et le vectoriseur CountVectorizer associé
(`.joblib`).

## Utiliser un modèle entraîné pour la prédiction

```python
from pathlib import Path
import joblib
from tensorflow.keras.models import load_model

from src.train_bow_ann import predict_text

model = load_model(Path("models/modele_trained.keras"))
vectorizer = joblib.load(Path("models/count_vectorizer.joblib"))
label, confidence = predict_text("Your text here", model, vectorizer)
print(label, confidence)
```

Vous pouvez associer l'identifiant numérique retourné à son émotion en vous référant à
la table précédente.

## Notebooks

Les notebooks conservés dans `notebooks/` retraçent l'exploration initiale :

- Comparaison de vecteurs de caractéristiques (Bag-of-Words, TF-IDF, Word Embeddings) ;
- Expériences avec des réseaux de neurones, arbres de décision et forêts aléatoires ;
- Prétraitements spécifiques (suppression des stop words, équilibrage des classes, etc.).

> 💡 **Astuce** : plusieurs notebooks ont été rédigés avant la réorganisation du dépôt et
> accèdent encore au fichier `text.csv` à la racine. Remplacez le chemin par `data/text.csv`
> lors de leur exécution.

## Documentation complémentaire

Le dossier [`docs/`](docs/) contient un résumé détaillé du projet et des pistes pour
reproduire les expériences historiques.

N'hésitez pas à ouvrir une issue ou une pull request si vous souhaitez contribuer
à l'amélioration du projet.
