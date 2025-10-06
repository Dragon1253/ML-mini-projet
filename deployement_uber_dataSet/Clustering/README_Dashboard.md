# 🚖 Dashboard Uber Clustering Analysis

## Description

Ce dashboard web interactif présente l'analyse des clusters Uber basée sur les données de géolocalisation des courses à New York. Il utilise l'algorithme HDBSCAN pour identifier les zones de forte demande et fournit des insights spatio-temporels complets.

## Structure du Dashboard

### 1. 🗺️ Carte Interactive des Clusters

- **Fonctionnalité principale** : Visualisation géographique des zones de demande
- **Caractéristiques** :
  - Clusters colorés distinctement
  - Points cliquables avec popups informatifs
  - Légende des couleurs par cluster
  - Zoom et navigation interactifs

### 2. 📊 Panneau de Contrôle

- **Filtres disponibles** :
  - Heure (0-23h)
  - Jour du mois (1-31)
  - Saison (Hiver, Printemps, Été, Automne)
- **Mise à jour en temps réel** de tous les graphiques et tableaux

### 3. 📈 Métriques Principales

- Nombre total de trajets
- Nombre de clusters identifiés
- Revenu moyen par course
- Nombre moyen de passagers

### 4. 🔍 Tableau Résumé des Clusters

Table interactive avec :

- ID du cluster
- Nombre de trajets
- Revenu moyen
- Passagers moyens
- Heure de pic
- Jour le plus actif

### 5. 📊 Graphiques d'Analyse

- **Heatmap Clusters × Heures** : Distribution temporelle par zone
- **Graphique de Revenu** : Répartition des revenus par cluster
- **Analyse Temporelle** : Évolution des trajets par heure

### 6. 💡 Insights Automatiques

Génération automatique d'insights clés :

- Cluster dominant en nombre de trajets
- Zone avec le revenu moyen le plus élevé
- Heures et saisons de pointe

## Installation et Utilisation

### Prérequis

```bash
pip install flask pandas joblib hdbscan numpy
```

### Fichiers nécessaires

- `uber_cleaned_clustered.csv` : Dataset nettoyé et clusterisé
- `hdbscan_model.pkl` : Modèle HDBSCAN entraîné
- `uber_app.py` : Application Flask
- `templates/dashboard.html` : Interface utilisateur

### Lancement

```bash
python uber_app.py
```

Le dashboard sera accessible à l'adresse : `http://127.0.0.1:5000`

## API Endpoints

L'application expose plusieurs endpoints pour les données :

- `GET /` : Page principale du dashboard
- `GET /api/clusters-data` : Données des clusters pour la carte
- `GET /api/cluster-summary` : Résumé statistique par cluster
- `GET /api/heatmap-data` : Données pour la heatmap
- `GET /api/temporal-analysis` : Analyse temporelle
- `GET /api/economic-analysis` : Analyse économique
- `GET /api/insights` : Insights automatiques
- `GET /api/filtered-data` : Données filtrées selon les paramètres
- `POST /predict` : Prédiction de cluster pour nouvelles coordonnées

## Fonctionnalités Avancées

### Filtrage Dynamique

- Sélection d'heures spécifiques
- Filtrage par jour ou saison
- Mise à jour instantanée de tous les éléments

### Interactivité

- Popups informatifs sur la carte
- Graphiques responsifs
- Navigation fluide

### Insights Automatiques

- Détection automatique des patterns
- Recommandations basées sur les données
- Résumés textuels intelligibles

## Structure Technique

### Frontend

- **HTML5/CSS3** : Interface moderne et responsive
- **Bootstrap 5** : Framework CSS pour la mise en forme
- **Leaflet.js** : Carte interactive
- **Chart.js** : Graphiques dynamiques
- **JavaScript ES6** : Logique côté client

### Backend

- **Flask** : Framework web Python
- **Pandas** : Manipulation des données
- **HDBSCAN** : Algorithme de clustering
- **Joblib** : Sérialisation du modèle

## Personnalisation

### Couleurs des Clusters

Modifiez le tableau `clusterColors` dans le JavaScript pour changer les couleurs.

### Métriques

Ajoutez de nouvelles métriques en modifiant les endpoints API et le frontend.

### Filtres

Étendez les options de filtrage en ajoutant de nouveaux paramètres aux endpoints.

## Performance

- Échantillonnage automatique pour les gros datasets (5000 points max)
- Requêtes API optimisées
- Cache côté client pour une navigation fluide

## Support

Pour toute question ou problème, vérifiez :

1. Que tous les fichiers de données sont présents
2. Que les dépendances sont installées
3. Que le modèle HDBSCAN est compatible
4. Que Flask fonctionne en mode debug

## Évolutions Futures

- Intégration de nouvelles sources de données
- Prédictions en temps réel
- Export des rapports en PDF
- API REST complète
- Authentification utilisateur
