######################## IMPORTS #########################################

import streamlit as st
import pandas as pd
import bcrypt
import requests
from requests.auth import HTTPBasicAuth
import base64
import page_1

#ML
from sklearn.preprocessing import StandardScaler  #avec écart-type
from sklearn.preprocessing import MinMaxScaler    #sensible aux valeurs extrêmes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import NearestNeighbors

######################## IMPORTS CSV ####################################

df_titres = pd.read_csv('df_titres.csv')
df_names = pd.read_csv('df_names.csv')

df_acteurs = pd.read_csv('df_acteurs.csv')
df_realisateurs = pd.read_csv('df_realisateurs.csv')
df_liste_noms = pd.read_csv('df_liste_noms.csv')
df_fr = pd.read_csv('df_fr.csv')
df_etranger = pd.read_csv('df_etranger.csv')
df_genres = pd.read_csv('df_genres.csv')

######################## CREATION D'UNE LISTE QUI NOUS SERVIRA POUR LE TRI DES FILMS PAR GENRE ####################################

liste_genres = ['Action', 'Aventure', 'Animation', 'Biographie',
       'Comedie', 'Crime', 'Documentaire', 'Drame', 'Famille', 'Fantasy',
       'Histoire', 'Horreur', 'Musique', 'Musical', 'Mystère', 'News', 'Romance',
       'Sci-Fi', 'Sport', 'Thriller', 'Guerre', 'Western']

df_titres_sorted = df_titres.sort_values('titreVF')
liste_choix = df_titres_sorted['titreVF'].tolist()
df_names_sorted = df_names.sort_values('Nom')
liste_choix = liste_choix + df_names_sorted['Nom'].tolist()
liste_choix = liste_choix + liste_genres
liste_choix = sorted(liste_choix, key=str)

######################## pR2PARATION DES DONNEES EN LISTE ####################################

col_a_modif = ['liste_realisateurs', 'liste_acteurs']
for col in col_a_modif :
  df_titres[col]=df_titres[col].str.split(',')

col_a_modif2 = ['commeActeur', 'commeRealisateur','films_plus_connus']
for col in col_a_modif2 :
  df_names[col]=df_names[col].str.split(',')

######################## Définition des fonctions ###############################

# POUR OBTENIR L'ID DU FILM A PARTIR DE SON TITRE ----------------------------------------
def id_du_film(titre) : # renvoie le tconst d'un film à partir du titreVF
  if  titre in df_titres['titreVF'].tolist() :  # Vérification que le film est bien connu                                  
    return df_titres.loc[df_titres['titreVF'] == titre]['tconst'].iloc[0]
  elif  titre in df_titres['TitreOriginal'].tolist() :  # Vérification que le film est bien connu                                  
    return df_titres.loc[df_titres['TitreOriginal'] == titre]['tconst'].iloc[0]   
  else : 
     st.write("Ce film n'est pas connu, vérifiez l'orthographe") #Si le film n'est pas trouvé dans notre BDD

# ON TRI LE DATAFRAME POUR GARDER QUE LES FILMS POUVANT CORRESPONDRE ----------------------------------------
def df_tri(film) :   # renvoie le dataframe utile pour les voisins en fonction du tconst
  
  # Création et ajustement du modèle NearestNeighbors
  df_act = df_titres # Création d'un dataframe de travail pour cette fonction
  langue = df_act.loc[df_act['tconst'] == film]['original_language'].iloc[0] # On récupère la langue du film
  
  df_act['act_commun'] = 0 # Initialisation d'une liste pour compter les acteurs en commun
  df_act['real_commun'] = 0 # Initialisation d'une liste pour compter les réalisateurs en commun
  for act in df_act.loc[df_act['tconst']==film]['liste_acteurs'].iloc[0] : # On parcourt la liste des acteurs du film choisi
    for film_act in df_names.loc[df_names['personneID']== act]['commeActeur'].iloc[0] : # On parcourt la liste des films des acteurs
      df_act.loc[df_act['tconst'] == film_act, 'act_commun'] += 1 # Incrémentation du nombre d'acteur en commun
  
  for real in df_act.loc[df_act['tconst']==film]['liste_realisateurs'].iloc[0] : 
    for film_real in df_names.loc[df_names['personneID']== real]['commeRealisateur'].iloc[0] :
      df_act.loc[df_act['tconst'] == film_real, 'real_commun'] += 1
  
  df_langue = df_act.loc[df_act['original_language'] == langue][['tconst','anneeSortie','noteMoyenne', 'nbVotes', 'act_commun', 'real_commun']] # récupération des colonnes intéressantes pour les films dans la langue
  
  
   #Normalisation des données :
  scaler = StandardScaler()           # J'utilise StandardScaler pour qu'il soit moins sensible aux valeurs extrêmes
  scaled = scaler.fit_transform(df_langue[['anneeSortie','noteMoyenne', 'nbVotes','act_commun', 'real_commun']])
  df_scaled = pd.DataFrame(scaled, columns=['anneeSortie','noteMoyenne', 'nbVotes','act_commun', 'real_commun'])
  df_scaled['act_commun'] = df_scaled['act_commun']/4 # On ajuste le poids des différentes colonnes
  df_scaled['real_commun'] = df_scaled['real_commun']/5
  df_scaled['noteMoyenne'] = df_scaled['noteMoyenne']*6
  df_scaled['anneeSortie'] = df_scaled['anneeSortie']*3
  df_scaled['nbVotes'] = df_scaled['nbVotes']*3

  df_langue = pd.concat([df_langue['tconst'].reset_index(drop=True), df_scaled], axis=1)
  df = pd.merge(df_langue,            #création d'un df avec toutes les données pour le KNN
               df_genres,
               how = 'inner',
               on = 'tconst')
  df = df.drop(['Unnamed: 0'], axis = 1)  
  # tri sur les genres : suppression des films qui n'ont aucun genre en commun
  #obtenir la liste des genres du film :
  
  liste_genres = [genre for genre in ['Action', 'Aventure', 'Animation', 'Biographie','Comedie', 'Crime', 'Documentaire', 'Drame', 'Famille', 'Fantasy','Histoire', 'Horreur', 'Musique', 'Musical', 'Mystère', 'News', 'Romance','Sci-Fi', 'Sport', 'Thriller', 'Guerre', 'Western'] if df.loc[df['tconst']==film][genre].iloc[0] == 1]
  
  #on supprime tous les films qui n'ont pas de genre en commun et de genre important
  liste_genre_princ = ['Animation','Crime', 'Documentaire', 'Famille', 'Horreur', 'Guerre', 'Western', 'Sci-Fi']
  if len(liste_genres) == 1 :
    df = df.loc[~(df[liste_genres[0]] == 0)]
    
  elif len(liste_genres) == 2 :
    df = df.loc[~((df[liste_genres[0]] == 0) & (df[liste_genres[1]] == 0))]
    if liste_genres[0] in liste_genre_princ :
         df = df.loc[~(df[liste_genres[0]] == 0)]
    elif liste_genres[1] in liste_genre_princ :
         df = df.loc[~(df[liste_genres[1]] == 0)]
  else :
    df = df.loc[~((df[liste_genres[0]] == 0) & (df[liste_genres[1]] == 0) & (df[liste_genres[2]] == 0))]
    if liste_genres[0] in liste_genre_princ :
         df = df.loc[~(df[liste_genres[0]] == 0)]
    elif liste_genres[1] in liste_genre_princ :
         df = df.loc[~(df[liste_genres[1]] == 0)]
    elif liste_genres[2] in liste_genre_princ :
         df = df.loc[~(df[liste_genres[2]] == 0)]
  
  return df
  
#----------------------------------------

def suggestions(df, film) :

  array = df.iloc[:,1:].to_numpy()           # transformation des valeurs utiles en array (j'aurais aussi pu utiliser un .values)
  nn = NearestNeighbors(n_neighbors=11, metric='euclidean')         # pour récupérer 10 films voisins
  nn.fit(array)
  mon_film = df.loc[df['tconst']==film].iloc[:,1:].to_numpy()        
  distances, indices = nn.kneighbors(mon_film)     # on récupère les distances et les indices des pokémons les plus proches

  liste_distances = distances.tolist()     # transformation de l'array des distances en liste
  liste_indices = indices.tolist()         # transformation de l'array des indices en liste

  del liste_indices[0][0]                 # supprime le premier de la liste (qui est le film cible)
  liste_indices = liste_indices[0]        # transforme la liste de liste en liste simple
  liste_finale = []
    
  liste_finale = [df_titres.loc[df_titres['tconst']==df.iloc[i]['tconst']][['titreVF']].iloc[0].iloc[0] for i in liste_indices]
  
  liste_imdb = [df_titres.loc[df_titres['tconst']==df.iloc[i]['tconst']][['id']].iloc[0].iloc[0] for i in liste_indices]

  return liste_finale, liste_imdb

#----------------------------------------

def filmograhie (nom_acteur) :
    IDactor = df_names.loc[df_names['Nom']==nom_acteur]['personneID'].iloc[0]
    if df_names.loc[df_names['Nom']==nom_acteur]['commeActeur'].iloc[0] == 'pas_de_film' :
       liste_finale = []
       liste_imdb = []
    else :
      df_actor_choisi = df_acteurs.loc[df_acteurs['nconst'] == IDactor]['tconst']
      df = pd.merge(df_titres,df_actor_choisi, how = 'inner', on = 'tconst')
      df = df.sort_values('notePonderee', ascending = False).head(10)
      liste_finale = df['titreVF'].tolist()
      liste_imdb = df['id'].tolist()
    if len(liste_finale)<10 :
      if df_names.loc[df_names['Nom']==nom_acteur]['commeRealisateur'].iloc[0] == 'pas_de_film' :
        liste_real = []
      else :
         df_real_choisi = df_realisateurs.loc[df_realisateurs['nconst'] == IDactor]['tconst']
         df = pd.merge(df_titres,df_real_choisi, how = 'inner', on = 'tconst')
         df = df.sort_values('notePonderee', ascending = False).head(10-len(liste_finale))
         liste_real = df['titreVF'].tolist()
         liste_imdb_real  = df['id'].tolist()
         liste_finale = liste_finale + liste_real
         liste_imdb = liste_imdb + liste_imdb_real
    return liste_finale, liste_imdb

#----------------------------------------

def suggestion_genre (genre) :
    df = df_genres.loc[df_genres[genre] == 1]['tconst']
    df = pd.merge(df_titres, df, how = 'inner', on = 'tconst')
    df = df.sort_values('notePonderee', ascending = False).head(10)
    liste_finale = df['titreVF'].tolist()
    liste_imdb = df['id'].tolist()
    return liste_finale, liste_imdb

#----------------------------------------

def searchMovies(query):
    url = f"https://api.themoviedb.org/3/movie/{query}" #details"
    params = {
        "api_key": 'ca250ec1056f9553bacc5cb920800fec',
        "language": "fr-FR",
        "include_adult": False
    }
    response = requests.get(url, params=params)
    data = response.json()
    return f"https://image.tmdb.org/t/p/original{data['poster_path']}"

######################## TEST de L'AFFICHAGE



with st.sidebar:
        titre_test = st.selectbox("Entrez votre critère de suggestion : ", liste_choix, index = None )
        requete_trouvee = 0
if titre_test is not None :
  if titre_test in df_titres['titreVF'].tolist() : # On cherche si c'est un titre de film
    requete_trouvee = 1
    film = id_du_film(titre_test)
    df = df_tri(film)
    films_finaux, imdb = suggestions(df, film)
  

  elif titre_test in liste_genres: # On cherche si c'est genre
    requete_trouvee = 1
    films_finaux, imdb = suggestion_genre(titre_test)      

  elif titre_test in df_names['Nom'].tolist(): # On cherche si c'est un acteur ou un réalisateur
    requete_trouvee = 1
    films_finaux, imdb = filmograhie(titre_test) 

  if requete_trouvee == 1 : # Si on a trouvé un résultat à cette requête on les affiches
    col1, col2  = st.columns(2)
    if len(films_finaux) == 0 :
      st.write("Nous n'avons pas trouvé de film")
    with col1: 
        st.write(films_finaux[0])
        st.image(searchMovies(imdb[0]), use_column_width=True)
        if len(films_finaux) >= 3 : # Les if sont pour éviter les messages d'erreur si on n'a pas 10 films
            st.write(films_finaux[2])
            st.image(searchMovies(imdb[2]), use_column_width=True)
        if len(films_finaux) >= 5 :    
            st.write(films_finaux[4])
            st.image(searchMovies(imdb[4]), use_column_width=True)
        if len(films_finaux) >= 7 : 
            st.write(films_finaux[6])
            st.image(searchMovies(imdb[6]), use_column_width=True)
        if len(films_finaux) >= 9 : 
            st.write(films_finaux[8])
            st.image(searchMovies(imdb[8]), use_column_width=True)
    with col2: 
        st.write(films_finaux[1])
        st.image(searchMovies(imdb[1]), use_column_width=True)
        if len(films_finaux) >= 4 : 
            st.write(films_finaux[3])
            st.image(searchMovies(imdb[3]), use_column_width=True)
        if len(films_finaux) >= 6 : 
            st.write(films_finaux[5])
            st.image(searchMovies(imdb[5]), use_column_width=True)
        if len(films_finaux) >= 8 : 
            st.write(films_finaux[7])
            st.image(searchMovies(imdb[7]), use_column_width=True)
        if len(films_finaux) >= 10 : 
            st.write(films_finaux[9])
            st.image(searchMovies(imdb[9]), use_column_width=True)
  else : # On a rien trouvé qui correspond à la requête
    st.write("Nous n'avons pas trouvé de résultat à votre recherche")
     

