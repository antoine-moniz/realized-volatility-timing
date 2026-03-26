# Projet - Realized Volatility Timing (UKF + Heston)

## 1) Ce que fait le projet

Ce projet implemente une strategie de timing de volatilite pour comparer:

- une **baseline**: carry short-vol statique
- une **strategie dynamique**: carry modulee par un signal IV-RV

Objectif: verifier si l'information extraite d'un modele Heston filtre par UKF peut ameliorer le ratio de Sharpe par rapport a une exposition statique.

## 2) Idee methodologique

Le pipeline suit les etapes suivantes:

1. **Preparation des donnees**
	- extraction des donnees options/spot
	- construction d'un proxy de volatilite implicite

2. **Estimation de volatilite realisee latente**
	- calibration rolling des parametres Heston par MLE
	- filtrage UKF pour obtenir une estimation lisse de variance/volatilite

3. **Construction du signal**
	- spread: IV - RV estimee
	- normalisation en z-score sur fenetre glissante

4. **Regle d'allocation**
	- baseline: multiplicateur fixe
	- dynamique: multiplicateur regime-based selon le z-score

5. **Backtest et evaluation**
	- NAV, Sharpe, max drawdown, Calmar
	- comparaison baseline vs dynamique

## 3) Strategie finale retenue dans le rendu

Le notebook final retient une allocation regime-based calibree localement:

- `low = -1.5`
- `high = 0.25`
- `m_neg = 0.2`
- `m_mid = 1.0`
- `m_pos = 1.15`

Sur la periode principale 2021-2022 (SPY), cette configuration donne une amelioration marginale du Sharpe de la strategie dynamique par rapport a la baseline.

## 4) Structure du code

- `projet/src/config.py`: configurations et hyperparametres
- `projet/src/data_pipeline.py`: preparation des series marche (spot, rendements, IV proxy)
- `projet/src/ukf_heston.py`: filtre UKF scalaire pour variance latente
- `projet/src/estimation.py`: calibration rolling MLE + filtrage
- `projet/src/signal_allocation.py`: signal IV-RV et multiplicateurs d'allocation
- `projet/src/experiment.py`: orchestration end-to-end et sorties de resultats
- `projet/notebooks/rendu_projet_realized_vol_timing.ipynb`: notebook final de rendu

## 5) Donnees et perimetre

Le projet reutilise les loaders du package `investment_lab`.

Points importants:

- la disponibilite temporelle depend des fichiers options/rates locaux
- dans la configuration actuelle du projet, les donnees sont bornees a fin 2023

## 6) Execution recommandee

1. Ouvrir `projet/notebooks/rendu_projet_realized_vol_timing.ipynb`
2. Executer les cellules dans l'ordre
3. Lire:
	- `results['perf']` pour les metriques
	- les graphes signal/NAV
	- la section "Interpretation Finale"

## 7) Limites

- resultat sensible au regime de marche
- gain de Sharpe faible: preuve de concept, pas preuve definitive de robustesse
- validation hors echantillon et couts reels necessaires avant usage operationnel
