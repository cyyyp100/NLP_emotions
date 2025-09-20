# Rapport du projet "NLP Emotions"

Ce document résume la démarche suivie pour entraîner un modèle de classification des émotions
sur des tweets en anglais. Il complète le README en offrant davantage de contexte sur les
choix effectués au cours du projet original.

## Objectif

L'objectif est de prédire l'émotion dominante d'un tweet par l'intermédiaire d'un modèle
d'apprentissage supervisé. Les six émotions considérées sont : tristesse, joie, amour,
colère, peur et surprise.

## Jeu de données

Le projet s'appuie sur le jeu de données **Emotion** disponible sur Kaggle. Chaque
observation contient deux colonnes :

- `text` : le contenu textuel du tweet (en anglais) ;
- `label` : un entier compris entre 0 et 5 représentant l'émotion associée au tweet.

Les labels sont répartis de manière légèrement déséquilibrée, la classe « Surprise » étant
moins fréquente. Le dossier [`data/`](../data/) contient un extrait du jeu de données utilisé
pour la reproduction rapide des expériences.

## Approche

Plusieurs familles de modèles ont été explorées au cours du projet initial :

- **Représentations de type sac de mots** (Bag-of-Words) combinées à un perceptron
  multicouche, implémentées dans le script [`src/train_bow_ann.py`](../src/train_bow_ann.py) ;
- **TF-IDF** avec et sans suppression des stop words pour alimenter des réseaux de neurones ;
- **Word Embeddings** suivis d'un classifieur dense ;
- **Arbres de décision** et **forêts aléatoires** sur des représentations CountVectorizer.

Les notebooks présents dans [`notebooks/`](../notebooks/) contiennent les expérimentations
et comparaisons détaillées de ces variantes.

## Reproductibilité

Le dépôt a été réorganisé pour faciliter la prise en main :

- Les données sont regroupées dans `data/` ;
- Les modèles entraînés sont stockés dans `models/` ;
- Les scripts réutilisables vivent dans `src/` ;
- Les notebooks exploratoires sont conservés dans `notebooks/` ;
- La présente documentation est centralisée dans `docs/`.

Suivez les instructions du README pour recréer l'environnement Python, relancer
l'entraînement du modèle Bag-of-Words et évaluer les performances obtenues sur l'échantillon
de validation.
