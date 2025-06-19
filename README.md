# Guide d’utilisation du projet

**Auteur :** Planchon Romain

---
(le notebook qui reprend le fil du rapport est main.ipynb)
J'ai malheuresement oublier de mettre une seed de random donc les résultats que vous obtiendrez ne seront pas necessairement les mêmes que dans le rapport.

(Les dependances de packages python sont dans requirements.txt)

## À lire avant d’exécuter les main.ipynb

1. **Organisation des fichiers**  
   Placez `main.ipynb` au même niveau que le dossier `Dataset`, qui doit contenir les sous-dossiers `Test` et `Train`.


2. **Plan général de `main.ipynb`**  
   a. **Segmentation**  
        - La segmentation du ventricule gauche (LV) est effectuée en premier.  
        - Un nouveau dossier `SegTest` est créé et contient la même chose que Test avec cette fois une segmentation complète.  

        - Ce dossier servira de base pour la suite du traitement.

        - Pour voir les cas particulier de la segmentation (ceux du rapport) aller dans le fichier `pipeline_segmentation.ipyb` et décommenter les exemples

   b. **Extraction des features (py-radiomics)**  
        - Activez ou désactivez les classes de features souhaitées en commentant/décommentant les mênes lignes que ci-dessous :

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

   d. **BASELINE MODEL**  

   e. **BASELINE MODEL filtered** 
    (with mRmR shape feature filterfing) feature reduced to 15
    
    f. **SVM Sequential Feature Selector on 50 mRMR  issued of shape,firstorder,glcm and glrlm

---

> **Remarque :** Tous les autres notebooks du dossier peuvent également être utilisés librement pour compléter ou approfondir l’analyse. Le main n'est qu'une compilation de quelques pipelines.



