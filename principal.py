######################## IMPORTS #########################################

import streamlit as st
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
import base64

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
       'Histoire', 'Horreur', 'Musique', 'Musical', 'Mystere', 'News', 'Romance',
       'Sci-Fi', 'Sport', 'Thriller', 'Guerre', 'Western']


######################## PREPARATION DES DONNEES EN LISTE ####################################

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
     st.write("Ce film n'est pas connu, verifiez l'orthographe") #Si le film n'est pas trouvé dans notre BDD

# ON TRI LE DATAFRAME POUR NE GARDER QUE LES FILMS POUVANT CORRESPONDRE ----------------------------------------
def df_tri(film) :   # renvoie le dataframe utile pour les voisins en fonction du tconst
  
  # Création et ajustement du modèle NearestNeighbors
  df_act = df_titres # Création d'un dataframe de travail pour cette fonction
  langue = df_act.loc[df_act['tconst'] == film]['original_language'].iloc[0] # On récupère la langue du film
  
  df_act = df_act.loc[df_act['original_language'] == langue][['tconst','liste_acteurs','liste_realisateurs','anneeSortie','noteMoyenne', 'nbVotes']] # récupération des colonnes intéressantes pour les films dans la langue

  # création de deux colonnes : 'act_commun' et 'real_commun' pour compter le nombre d'acteurs / réalisateur communs entre le film et les autres films du df
  df_act['act_commun'] = 0 # Initialisation d'une liste pour compter les acteurs en commun
  df_act['real_commun'] = 0 # Initialisation d'une liste pour compter les réalisateurs en commun
  for act in df_act.loc[df_act['tconst']==film]['liste_acteurs'].iloc[0] : # On parcourt la liste des acteurs du film choisi
    for film_act in df_names.loc[df_names['personneID']== act]['commeActeur'].iloc[0] : # On parcourt la liste des films des acteurs
      df_act.loc[df_act['tconst'] == film_act, 'act_commun'] += 1 # Incrémentation du nombre d'acteurs en commun
  
  for real in df_act.loc[df_act['tconst']==film]['liste_realisateurs'].iloc[0] : 
    for film_real in df_names.loc[df_names['personneID']== real]['commeRealisateur'].iloc[0] :
      df_act.loc[df_act['tconst'] == film_real, 'real_commun'] += 1
  
  # on supprime les listes acteurs et réalisateurs qui ne sont plus nécessaires pour la suite :
  df_act = df_act[['tconst','anneeSortie','noteMoyenne', 'nbVotes', 'act_commun', 'real_commun']]
  
  #Normalisation des données :
  scaler = StandardScaler()           # J'utilise StandardScaler pour qu'il soit moins sensible aux valeurs extrêmes
  scaled = scaler.fit_transform(df_act[['anneeSortie','noteMoyenne', 'nbVotes','act_commun', 'real_commun']])
  df_scaled = pd.DataFrame(scaled, columns=['anneeSortie','noteMoyenne', 'nbVotes','act_commun', 'real_commun'])
  df_scaled['act_commun'] = df_scaled['act_commun']/4 # On ajuste le poids des différentes colonnes
  df_scaled['real_commun'] = df_scaled['real_commun']/5
  df_scaled['noteMoyenne'] = df_scaled['noteMoyenne']*6
  df_scaled['anneeSortie'] = df_scaled['anneeSortie']*3
  df_scaled['nbVotes'] = df_scaled['nbVotes']*3

  df_act = pd.concat([df_act['tconst'].reset_index(drop=True), df_scaled], axis=1)
  df = pd.merge(df_act,            #création d'un df avec toutes les données pour le KNN
               df_genres,
               how = 'inner',
               on = 'tconst')
  df = df.drop(['Unnamed: 0'], axis = 1)  

  # tri sur les genres : suppression des films qui n'ont aucun genre en commun
  
  #obtenir la liste des genres du film :
  genres_film = [genre for genre in liste_genres if df.loc[df['tconst']==film][genre].iloc[0] == 1]
  
  #on supprime tous les films qui n'ont pas de genre en commun et de genre important
  liste_genre_princ = ['Comedie','Animation','Crime', 'Documentaire', 'Famille', 'Horreur', 'Guerre', 'Western', 'Sci-Fi']
  
  if len(genres_film) == 1 :                    # si le film n'a qu'un genre, on gare uniquement les films qui contiennent ce genre
    df = df.loc[~(df[genres_film[0]] == 0)]
    
  elif len(genres_film) == 2 :
    df = df.loc[~((df[genres_film[0]] == 0) & (df[genres_film[1]] == 0))]     # si le film a deux genres, on supprime tous les films qui n'ont aucun de ces deux genres
    if genres_film[0] in liste_genre_princ :          # si le film a un genre parmis les genres principaux, on supprime tous les films n'ayant pas ce genre principal 
         df = df.loc[~(df[genres_film[0]] == 0)]
    elif genres_film[1] in liste_genre_princ :
         df = df.loc[~(df[genres_film[1]] == 0)]
  else :
    df = df.loc[~((df[genres_film[0]] == 0) & (df[genres_film[1]] == 0) & (df[genres_film[2]] == 0))]   # si le film a 3 genres, on supprime tous les films qui n'ont aucun de ces 3 genres
    if genres_film[0] in liste_genre_princ :         # si le film a un genre parmis les genres principaux, on supprime tous les films n'ayant pas ce genre principal 
         df = df.loc[~(df[genres_film[0]] == 0)]
    elif genres_film[1] in liste_genre_princ :
         df = df.loc[~(df[liste_genres[1]] == 0)]
    elif genres_film[2] in liste_genre_princ :
         df = df.loc[~(df[genres_film[2]] == 0)]
  
  return df
  
#----------------------------------------



def suggestions(df, film) :

  array = df.iloc[:,1:].to_numpy()           # transformation des valeurs utiles en array (j'aurais aussi pu utiliser un .values)
  nn = NearestNeighbors(n_neighbors=11, metric='euclidean')         # pour récupérer 10 films voisins
  nn.fit(array)
  mon_film = df.loc[df['tconst']==film].iloc[:,1:].to_numpy()        
  distances, indices = nn.kneighbors(mon_film)     # on récupère les distances et les indices des films les plus proches

  liste_indices = indices.tolist()         # transformation de l'array des indices en liste

  del liste_indices[0][0]                 # supprime le premier de la liste (qui est le film cible)
  liste_indices = liste_indices[0]        # transforme la liste de liste en liste simple
  liste_finale = []
    
  liste_finale = [df_titres.loc[df_titres['tconst']==df.iloc[i]['tconst']][['titreVF']].iloc[0].iloc[0] for i in liste_indices]  # liste des titres de films des 10 plus proches voisins
  
  liste_tmdb = [df_titres.loc[df_titres['tconst']==df.iloc[i]['tconst']][['id']].iloc[0].iloc[0] for i in liste_indices]     # liste des id TMDB des 10 plus proches voisins (pour affichage des affiches de films)

  return liste_finale, liste_tmdb

#----------------------------------------

def filmograhie (nom_acteur) :              # renvoie la liste des 10 films ayant la meilleure note pondérée de l'acteur ou du réalisateur.
    IDactor = df_names.loc[df_names['Nom']==nom_acteur]['personneID'].iloc[0]
    if df_names.loc[df_names['Nom']==nom_acteur]['commeActeur'].iloc[0] == 'pas_de_film' :
       liste_finale = []
       liste_tmdb = []
    else :
      df_actor_choisi = df_acteurs.loc[df_acteurs['nconst'] == IDactor]['tconst']
      df = pd.merge(df_titres,df_actor_choisi, how = 'inner', on = 'tconst')
      df = df.sort_values('notePonderee', ascending = False).head(10)
      liste_finale = df['titreVF'].tolist()
      liste_tmdb = df['id'].tolist()
    if len(liste_finale)<10 :
      if df_names.loc[df_names['Nom']==nom_acteur]['commeRealisateur'].iloc[0] == 'pas_de_film' :
        liste_real = []
      else :
         df_real_choisi = df_realisateurs.loc[df_realisateurs['nconst'] == IDactor]['tconst']
         df = pd.merge(df_titres,df_real_choisi, how = 'inner', on = 'tconst')
         df = df.sort_values('notePonderee', ascending = False).head(10-len(liste_finale))
         liste_real = df['titreVF'].tolist()
         liste_tmdb_real  = df['id'].tolist()
         liste_finale = liste_finale + liste_real
         liste_tmdb = liste_tmdb + liste_tmdb_real
    return liste_finale, liste_tmdb

#----------------------------------------

def suggestion_genre (genre) :
    df = df_genres.loc[df_genres[genre] == 1]['tconst']
    df = pd.merge(df_titres, df, how = 'inner', on = 'tconst')
    df = df.sort_values('notePonderee', ascending = False).head(10)
    liste_finale = df['titreVF'].tolist()
    liste_tmdb = df['id'].tolist()
    return liste_finale, liste_tmdb

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

df_titres_sorted = df_titres.sort_values('titreVF')
df_names_sorted = df_names.sort_values('Nom')


with st.sidebar:
        type_choix = st.selectbox("Quel type de recherche voulez-vous faire ?",['par film','par acteur','par réalisateur'])
        if type_choix == 'par film' :
          liste_choix = df_titres_sorted['titreVF'].tolist()
          phrase = "Entrez le nom d'un film que vous avez aimé : "
        elif type_choix == 'par acteur' :
          liste_choix = df_names.loc[df_names['commeActeur']!='pas de film'].sort_values('Nom').tolist()
          phrase = "Entrez le nom de l'acteur : "
        else : 
          liste_choix = df_names.loc[df_names['commeRealisateur']!='pas de film'].sort_values('Nom').tolist()
          phrase = "Entrez le nom du réalisateur : "
  
        liste_choix = sorted(liste_choix, key=str)
        titre_test = st.selectbox(phrase, liste_choix, index = None )

        requete_trouvee = 0
        
        #genre_choisi = st.radio("Choix du genre du film",liste_genres)

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

  if requete_trouvee == 1 : # Si on a trouvé un résultat à cette requête on les affiche
    col1, col2  = st.columns(2)
    if len(films_finaux) == 0 :
      st.write("Nous n'avons pas trouvé de film")
    with col1: 
        st.write(films_finaux[0])
        st.image(searchMovies(imdb[0]), use_container_width=True)
        if len(films_finaux) >= 3 : # Les if sont pour éviter les messages d'erreur si on n'a pas 10 films
            st.write(films_finaux[2])
            st.image(searchMovies(imdb[2]), use_container_width=True)
        if len(films_finaux) >= 5 :    
            st.write(films_finaux[4])
            st.image(searchMovies(imdb[4]), use_container_width=True)
        if len(films_finaux) >= 7 : 
            st.write(films_finaux[6])
            st.image(searchMovies(imdb[6]), use_container_width=True)
        if len(films_finaux) >= 9 : 
            st.write(films_finaux[8])
            st.image(searchMovies(imdb[8]), use_container_width=True)
    with col2: 
        st.write(films_finaux[1])
        st.image(searchMovies(imdb[1]), use_container_width=True)
        if len(films_finaux) >= 4 : 
            st.write(films_finaux[3])
            st.image(searchMovies(imdb[3]), use_container_width=True)
        if len(films_finaux) >= 6 : 
            st.write(films_finaux[5])
            st.image(searchMovies(imdb[5]), use_container_width=True)
        if len(films_finaux) >= 8 : 
            st.write(films_finaux[7])
            st.image(searchMovies(imdb[7]), use_container_width=True)
        if len(films_finaux) >= 10 : 
            st.write(films_finaux[9])
            st.image(searchMovies(imdb[9]), use_container_width=True)
  else : # On a rien trouvé qui correspond à la requête
    st.write("Nous n'avons pas trouvé de résultat à votre recherche")
     

