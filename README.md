# Procom_31_4G_classif

## Mesures #

Le fichier `journal.txt` donne plus de détails sur le déroulé de l'acquisition pour chaque fichier `Trace_XX.csv`.

- Le dossier 0_Process_and_Label_data contient 2 scripts permettant de pré-traiter et labelliser les données 4G et WiFi (fichiers csv des données décodées sous la forme `TraceYYYYMMDD_HHMMSS_with_id.csv` pour la 4G, fichiers de la forme `WifiYYYYMMDD_*.csv` pour les captures WiFi). Le dataset complet est disponible sur Kaggle : https://www.kaggle.com/datasets/testprocom/procom-4g-classification.


## Organisation du Github #

Les mesures sont importés dans le dossier Mesures. Ce dossier est séparé en 2 dossier distincts : Train et test afin de séparer les datasets.

Le dossier 1_Format_donnees contient les notebooks que nous avons mis en place afin de réaliser notre première expérience concernant les formats des données. Il contient aussi un fichier aggregation.py qui contient un ensemble de fonctions utilisés dans nos expériences. Le but étant d'avoir des notebook lisibles.

Le dossier 3_Classification_Model contient les notebooks relatif à la comparaison des modèles de classification.

Le dossier Experimental_Analysis contient des notebooks où nous avons mis en place nos premières analyses et modèles expérimentaux.

Le fichier evaluate.py contient des scripts permettant l'évaluation d'un modèle au vu des métriques que nous avons définit.



