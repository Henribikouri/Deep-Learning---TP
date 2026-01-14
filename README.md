# 🚀 Deep Learning Engineering - ENSPY 5GI/M2

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework: TensorFlow/Keras](https://img.shields.io/badge/Framework-TensorFlow-orange.svg)](https://tensorflow.org)
[![MLOps: MLflow](https://img.shields.io/badge/MLOps-MLflow-blueviolet.svg)](https://mlflow.org)
[![Container: Docker](https://img.shields.io/badge/Container-Docker-blue.svg)](https://www.docker.com/)

Ce dépôt rassemble l'ensemble des travaux pratiques (TP 1 à 4) du module **Deep Learning Engineering**. L'objectif est de démontrer une maîtrise complète du cycle de vie des modèles, de la recherche théorique au déploiement industriel.



---

## 📂 Structure du Projet

Le projet est organisé par TP, chacun étant autonome avec son propre environnement :

* **`TP1 &2 (Foundations and Optimization)/`** : Classification MNIST, API Flask & Conteneurisation Docker and Diagnostic Biais/Variance, Régularisation (L2, Dropout) & Optimiseurs.
* **`TP3_CNN_Vision/`** : Réseaux Convolutionnels sur CIFAR-10, ResNet & Transfert de Style.
* **`TP4_Segmentation_3D/`** : Architecture U-Net médicale, métriques Dice/IoU & Conv3D.

---

## 🛠️ Installation et Utilisation

### 1. Cloner le projet
```bash
git clone [https://github.com/Henribikouri/Deep-Learning---TP](https://github.com/Henribikouri/Deep-Learning---TP)
cd Deep-Learning---TP
```
### 2. Exécuter un TP (Exemple TP2)
Chaque dossier contient son fichier requirements.txt pour isoler les dépendances.

```bash
cd TP1 &2 (Foundations and Optimization)
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate sous Windows
pip install -r requirements.txt
python train_model.py
```


### 3. Suivi avec MLflow
Pour visualiser les performances et les hyperparamètres enregistrés :

```bash
mlflow ui
```
nsuite, ouvrez http://localhost:5000 dans votre navigateur.

##👨‍💻 Auteur
  HENRI BIKURI - Master of Engineering Génie Informatique (5GI)

Institution : École Nationale Supérieure Polytechnique de Yaoundé (ENSPY)
SITE WEB : https://henribikouri.netlify.app/
Superviseurs : Dr. Fippo Fitime, M. Tinku, M. Sonfack.
