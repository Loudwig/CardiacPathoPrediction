
### 1. Trouver des features cohérents et les caluler.
    - volume des deux ventricules et mycordium.
        
### 2. Segmentation ventricule gauche pour le test.

Empiriquement on voit que le ventricule gauche est situé au milieu de la myocardium. On va donc prendre avantage que dans le test la mycordium est déjà segmenter.  

-> on va déja vérifier la donné. 



Ok 

###  Les features. 

1) volume de tout + Masse + taille. + dif temporel. 

Ce que ne peut pas faire un random forest c'est rapport donc il faut que dans features il y ait des rapport. On va faire ça
J'ai mis au début la difference entre les deux timings mais en fait un random forest peut trouver ça aussi (a-b > 4 ) random forest peut faire a> 6 b<2 donc pas besoin je les enlèves 
{'classifier__max_depth': 10, 'classifier__max_features': 'sqrt', 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 2, 'classifier__n_estimators': 100}
0.85 0.07071067811865474
1.0 0.0

I added the ratios. Now I have got around 30 features.

### 3. Le point majeur de ce challenge. Il n'y a pas beacoup de données. 

I will try data augmentation to see the results.

I know what i am doing wrong... In my cross val as i have data and a bit of noisy data that is almost the same the score don't really mean something as it is like trainning on it..