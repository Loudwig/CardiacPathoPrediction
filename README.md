# Guide d’utilisation du projet

**Étudiant :** Planchon Romain

---

## À lire avant d’exécuter les scripts

1. **Organisation des fichiers**  
   Placez `main.ipynb` au même niveau que le dossier `Dataset`, qui doit contenir les sous-dossiers `Test` et `Train`.

2. **Fonctionnement général de `main.ipynb`**  
   a. **Segmentation**  
   - La segmentation du ventricule gauche (LV) est effectuée en premier.  
   - Un nouveau dossier `SegTest` est créé et contient la même chose que Test avec cette fois une segmentation complète.  

   - Ce dossier servira de base pour la suite du traitement.
   
   - Pour voir les cas particulier de la segmentation aller dans le fichier `segmentation.py` et décommenter les exemples

   b. **Extraction des features (py-radiomics)**  
   - Activez ou désactivez les classes de features souhaitées en commentant/décommentant les lignes ci-dessous :

     ```python
     # Features à extraire :
     extractor.enableFeatureClassByName('shape')      # Forme (volume, sphéricité…)
     extractor.enableFeatureClassByName('firstorder') # Statistiques de premier ordre
     extractor.enableFeatureClassByName('glcm')       # Gray Level Co-occurrence Matrix
     extractor.enableFeatureClassByName('glrlm')      # Gray Level Run Length Matrix
     ```

   - Les jeux de données extraits seront sauvegardés dans :
     - `data/shape_glcm_features` si vous avez sélectionné `shape` et `glcm`
     - `data/shape_firstorder_glrlm_features` si vous avez désélectionné `glrlm`

   - Ces chemins sont automatiquement utilisés pour charger les jeux de données.

   c. **Réduction des features**  
   - Seuil de variance  
   - Seuil de corrélation  
   - Sélection mRMR (par défaut 50 features)

   d. **Modèle de référence (baseline)**  
   - Entraînement et évaluation du modèle de base sur les données réduites.

---

> **Remarque :** Tous les autres notebooks du dossier peuvent également être utilisés librement pour compléter ou approfondir l’analyse.  
