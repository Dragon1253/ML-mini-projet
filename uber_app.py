from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from hdbscan import prediction
import json

# Charger le modèle et les données
hdb = joblib.load("hdbscan_model.pkl")
df = pd.read_csv("uber_cleaned_clustered.csv")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('dashboard.html')



@app.route('/api/clusters-data')
def get_clusters_data():
    """Retourne les données des clusters pour la carte"""
    # Échantillonner les données pour éviter la surcharge
    sample_size = 5000
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    clusters_data = []
    for cluster_id in sorted(df_sample['cluster'].unique()):
        cluster_data = df_sample[df_sample['cluster'] == cluster_id]
        
        clusters_data.append({
            'cluster_id': int(cluster_id),
            'points': [
                {
                    'lat': float(row['pickup_latitude']),
                    'lng': float(row['pickup_longitude']),
                    'hour': int(row['hour']),
                    'month': int(row['month']),
                    'season': str(row['season']),
                    'fare': float(row['fare_amount']),
                    'passengers': int(row['passenger_count'])
                } for _, row in cluster_data.iterrows()
            ]
        })
    
    return jsonify(clusters_data)

@app.route('/api/cluster-summary')
def get_cluster_summary():
    """Retourne le résumé statistique de chaque cluster"""
    summary = df.groupby('cluster').agg({
        'pickup_latitude': 'count',  # nombre de trajets
        'fare_amount': 'mean',       # revenu moyen
        'passenger_count': 'mean',   # passagers moyens
        'hour': lambda x: x.mode().iloc[0],  # heure de pic
        'day': lambda x: x.mode().iloc[0]    # jour le plus actif
    }).reset_index()
    
    summary.columns = ['cluster_id', 'total_trips', 'avg_revenue', 'avg_passengers', 'peak_hour', 'most_active_day']
    
    return jsonify(summary.to_dict('records'))

@app.route('/api/heatmap-data')
def get_heatmap_data():
    """Retourne les données pour la heatmap cluster × heures"""
    # Récupérer les filtres
    hour = request.args.get('hour')
    day = request.args.get('day')
    season = request.args.get('season')
    
    filtered_df = df.copy()
    
    # Appliquer les filtres
    if hour:
        filtered_df = filtered_df[filtered_df['hour'] == int(hour)]
    if day:
        filtered_df = filtered_df[filtered_df['day'] == int(day)]
    if season:
        filtered_df = filtered_df[filtered_df['season'] == season]
    
    heatmap_data = filtered_df.groupby(['cluster', 'hour']).size().reset_index(name='trips_count')
    
    # Créer une matrice pivot
    pivot = heatmap_data.pivot(index='cluster', columns='hour', values='trips_count')
    pivot = pivot.fillna(0)
    
    # Convertir en format adapté pour le frontend
    result = []
    for cluster in pivot.index:
        for hour_col in pivot.columns:
            result.append({
                'cluster': int(cluster),
                'hour': int(hour_col),
                'trips': int(pivot.loc[cluster, hour_col])
            })
    
    return jsonify(result)

@app.route('/api/economic-analysis')
def get_economic_analysis():
    """Retourne l'analyse économique par cluster"""
    # Récupérer les filtres
    hour = request.args.get('hour')
    day = request.args.get('day')
    season = request.args.get('season')
    
    filtered_df = df.copy()
    
    # Appliquer les filtres
    if hour:
        filtered_df = filtered_df[filtered_df['hour'] == int(hour)]
    if day:
        filtered_df = filtered_df[filtered_df['day'] == int(day)]
    if season:
        filtered_df = filtered_df[filtered_df['season'] == season]
    
    economic_data = filtered_df.groupby('cluster').agg({
        'fare_amount': ['sum', 'mean', 'count']
    }).reset_index()
    
    economic_data.columns = ['cluster', 'total_revenue', 'avg_revenue', 'total_trips']
    economic_data['revenue_percentage'] = (economic_data['total_revenue'] / economic_data['total_revenue'].sum() * 100)
    
    return jsonify(economic_data.to_dict('records'))

@app.route('/api/temporal-analysis')
def get_temporal_analysis():
    """Retourne l'analyse temporelle par cluster"""
    # Récupérer les filtres
    hour = request.args.get('hour')
    day = request.args.get('day')
    season = request.args.get('season')
    
    filtered_df = df.copy()
    
    # Appliquer les filtres
    if hour:
        filtered_df = filtered_df[filtered_df['hour'] == int(hour)]
    if day:
        filtered_df = filtered_df[filtered_df['day'] == int(day)]
    if season:
        filtered_df = filtered_df[filtered_df['season'] == season]
    
    temporal_data = filtered_df.groupby(['cluster', 'hour']).agg({
        'pickup_latitude': 'count',
        'fare_amount': 'mean'
    }).reset_index()
    
    temporal_data.columns = ['cluster', 'hour', 'trips', 'avg_fare']
    
    return jsonify(temporal_data.to_dict('records'))

@app.route('/api/insights')
def get_insights():
    """Génère des insights automatiques"""
    insights = []
    
    # Cluster avec le plus de trajets
    top_cluster = df.groupby('cluster').size().idxmax()
    top_cluster_trips = df.groupby('cluster').size().max()
    total_trips = len(df)
    percentage = (top_cluster_trips / total_trips) * 100
    
    insights.append(f"Le Cluster {top_cluster} représente {percentage:.1f}% des trajets avec {top_cluster_trips:,} courses.")
    
    # Cluster avec le revenu moyen le plus élevé
    revenue_by_cluster = df.groupby('cluster')['fare_amount'].mean()
    highest_revenue_cluster = revenue_by_cluster.idxmax()
    highest_revenue = revenue_by_cluster.max()
    
    insights.append(f"Le Cluster {highest_revenue_cluster} génère le revenu moyen le plus élevé ({highest_revenue:.1f}$).")
    
    # Heure de pic globale
    peak_hour = df.groupby('hour').size().idxmax()
    insights.append(f"L'heure de pointe globale est {peak_hour}h.")
    
    # Saison la plus active
    peak_season = df.groupby('season').size().idxmax()
    insights.append(f"La saison la plus active est {peak_season}.")
    
    return jsonify(insights)

@app.route('/api/filtered-data')
def get_filtered_data():
    """Retourne les données filtrées selon les paramètres"""
    hour = request.args.get('hour')
    day = request.args.get('day')
    season = request.args.get('season')
    
    filtered_df = df.copy()
    
    # Appliquer les filtres
    if hour:
        filtered_df = filtered_df[filtered_df['hour'] == int(hour)]
    if day:
        filtered_df = filtered_df[filtered_df['day'] == int(day)]
    if season:
        filtered_df = filtered_df[filtered_df['season'] == season]
    
    # Calculer le résumé selon les filtres appliqués
    if len(filtered_df) == 0:
        return jsonify({
            'summary': [],
            'clusters_data': [],
            'insights': [],
            'total_filtered': 0
        })
    
    # Résumé adaptatif selon les filtres
    summary_data = []
    
    for cluster_id in sorted(filtered_df['cluster'].unique()):
        cluster_data = filtered_df[filtered_df['cluster'] == cluster_id]
        
        if len(cluster_data) == 0:
            continue
            
        summary_row = {
            'cluster_id': int(cluster_id),
            'total_trips': len(cluster_data),
            'avg_revenue': float(cluster_data['fare_amount'].mean()),
            'avg_passengers': float(cluster_data['passenger_count'].mean())
        }
        
        # Colonnes adaptatives selon les filtres
        if day:
            # Si jour spécifié → montrer l'heure de pic pour ce jour
            peak_hour = cluster_data.groupby('hour').size().idxmax() if len(cluster_data) > 0 else 0
            summary_row['peak_hour'] = int(peak_hour)
            summary_row['most_active_day'] = int(day)  # Jour fixé par le filtre
        elif hour:
            # Si heure spécifiée → montrer le jour de pic pour cette heure
            peak_day = cluster_data.groupby('day').size().idxmax() if len(cluster_data) > 0 else 1
            summary_row['peak_hour'] = int(hour)  # Heure fixée par le filtre
            summary_row['most_active_day'] = int(peak_day)
        elif season:
            # Si saison spécifiée → montrer l'heure et jour de pic pour cette saison
            peak_hour = cluster_data.groupby('hour').size().idxmax() if len(cluster_data) > 0 else 0
            peak_day = cluster_data.groupby('day').size().idxmax() if len(cluster_data) > 0 else 1
            summary_row['peak_hour'] = int(peak_hour)
            summary_row['most_active_day'] = int(peak_day)
        else:
            # Aucun filtre → comportement normal
            peak_hour = cluster_data.groupby('hour').size().idxmax() if len(cluster_data) > 0 else 0
            peak_day = cluster_data.groupby('day').size().idxmax() if len(cluster_data) > 0 else 1
            summary_row['peak_hour'] = int(peak_hour)
            summary_row['most_active_day'] = int(peak_day)
        
        summary_data.append(summary_row)
    
    # Données pour la carte (échantillonnées)
    sample_size = 5000
    if len(filtered_df) > sample_size:
        df_sample = filtered_df.sample(n=sample_size, random_state=42)
    else:
        df_sample = filtered_df
    
    clusters_data = []
    for cluster_id in sorted(df_sample['cluster'].unique()):
        cluster_points = df_sample[df_sample['cluster'] == cluster_id]
        
        clusters_data.append({
            'cluster_id': int(cluster_id),
            'points': [
                {
                    'lat': float(row['pickup_latitude']),
                    'lng': float(row['pickup_longitude']),
                    'hour': int(row['hour']),
                    'month': int(row['month']),
                    'season': str(row['season']),
                    'fare': float(row['fare_amount']),
                    'passengers': int(row['passenger_count'])
                } for _, row in cluster_points.iterrows()
            ]
        })
    
    # Insights filtrés
    insights = []
    if len(filtered_df) > 0:
        # Cluster avec le plus de trajets (filtré)
        top_cluster = filtered_df.groupby('cluster').size().idxmax()
        top_cluster_trips = filtered_df.groupby('cluster').size().max()
        total_trips = len(filtered_df)
        percentage = (top_cluster_trips / total_trips) * 100
        
        filter_context = ""
        if hour: filter_context += f" à {hour}h"
        if day: filter_context += f" le jour {day}"
        if season: filter_context += f" en {season}"
        
        insights.append(f"Le Cluster {top_cluster} représente {percentage:.1f}% des trajets{filter_context} avec {top_cluster_trips:,} courses.")
        
        # Cluster avec le revenu moyen le plus élevé (filtré)
        revenue_by_cluster = filtered_df.groupby('cluster')['fare_amount'].mean()
        highest_revenue_cluster = revenue_by_cluster.idxmax()
        highest_revenue = revenue_by_cluster.max()
        
        insights.append(f"Le Cluster {highest_revenue_cluster} génère le revenu moyen le plus élevé{filter_context} ({highest_revenue:.1f}$).")
        
        # Contexte temporel
        if not hour:
            peak_hour_filtered = filtered_df.groupby('hour').size().idxmax()
            insights.append(f"L'heure de pointe{filter_context} est {peak_hour_filtered}h.")
        
        if not day:
            peak_day_filtered = filtered_df.groupby('day').size().idxmax()
            insights.append(f"Le jour le plus actif{filter_context} est le {peak_day_filtered}.")
    
    return jsonify({
        'summary': summary_data,
        'clusters_data': clusters_data,
        'insights': insights,
        'total_filtered': len(filtered_df)
    })

# Endpoint prédiction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    lat = data['latitude']
    lon = data['longitude']
    # Convertir en radians car HDBSCAN a été entraîné comme ça
    coords_rad = np.radians([[lat, lon]])
    # Utiliser la fonction prediction.approximate_predict
    cluster_pred = prediction.approximate_predict(hdb, coords_rad)[0]

    return jsonify({"cluster": int(cluster_pred)})

if __name__ == "__main__":
    app.run(debug=True)
