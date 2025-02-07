######################## IMPORTS #########################################

import streamlit as st
import pandas as pd
import streamlit_authenticator as stauth
from streamlit_option_menu import option_menu
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

df_genres = df_genres[['tconst', 'Action', 'Adventure', 'Animation', 'Biography',
       'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
       'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
       'Sci-Fi', 'Sport', 'Thriller', 'War', 'Western']]

######################## Définition des fonctions ###############################

def id_du_film(titre) : # renvoie le tconst d'un film à partir du titreVF
  if  titre in df_titres['titreVF'].tolist() :  # Vérification que le film est bien connu                                  
    film_avec_ce_nom = df_titres.loc[df_titres['titreVF'] == titre]['tconst']     # un df avec tous les films qui portent ce nom (ex : 'À couteaux tirés' )
    return df_titres.loc[df_titres['titreVF'] == titre]['tconst'].iloc[0]  
  else : 
     st.write("Ce film n'est pas connu, vérifiez l'orthographe") #Si le film n'est pas trouvé dans notre BDD
#----------------------------------------

def df_tri(film) :   # renvoie le dataframe utile pour les voisins en fonction du tconst
  # Création et ajustement du modèle NearestNeighbors
  
  langue = df_titres.loc[df_titres['tconst'] == film]['original_language'].iloc[0]
  df_langue = df_titres.loc[df_titres['original_language'] == langue][['tconst','anneeSortie','noteMoyenne', 'nbVotes']]      # récupération des colonnes intéressantes pour les films dans la langue
  
  #Normalisation des données :
  scaler = StandardScaler()           # J'utilise StandardScaler pour qu'il soit moins sensible aux valeurs extrêmes
  scaled = scaler.fit_transform(df_langue[['anneeSortie','noteMoyenne', 'nbVotes']])
  df_scaled = pd.DataFrame(scaled, columns=['anneeSortie','noteMoyenne', 'nbVotes'])
  df_langue = pd.concat([df_langue['tconst'].reset_index(drop=True), df_scaled], axis=1)
  
  df = pd.merge(df_langue,            #création d'un df avec toutes les données pour le KNN
               df_genres,
               how = 'inner',
               on = 'tconst')
  df['nbVotes'] = df['nbVotes'].apply(lambda x : 2*x)
  
  # tri sur les genres : suppression des films qui n'ont aucun genre en commun
  #obtenir la liste des genres du film :
  liste_genres = [genre for genre in ['Action', 'Adventure', 'Animation', 'Biography','Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy','History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance','Sci-Fi', 'Sport', 'Thriller', 'War', 'Western'] if df.loc[df['tconst']==film][genre].iloc[0] == 1]
  
  #on supprime tous les films qui n'ont pas de genre en commun et de genre important
  liste_genre_princ = ['Animation','Crime', 'Documentary', 'Family', 'Horror', 'War', 'Western', 'Sci-Fi']
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
  
  return liste_finale

#----------------------------------------

def filmograhie (nom_acteur) :
    IDactor = df_names.loc[df_names['Nom']==nom_acteur]['personneID'].iloc[0]
    if df_names.loc[df_names['Nom']==nom_acteur]['commeActeur'].iloc[0] == 'pas_de_film' :
       liste_finale = [] 
    else :
      df_actor_choisi = df_acteurs.loc[df_acteurs['nconst'] == IDactor]['tconst']
      df = pd.merge(df_titres,df_actor_choisi, how = 'inner', on = 'tconst')
      df = df.sort_values('notePonderee', ascending = False).head(10)
      liste_finale = df['titreVF'].tolist()
      
    if len(liste_finale)<10 :
      if df_names.loc[df_names['Nom']==nom_acteur]['commeRealisateur'].iloc[0] == 'pas_de_film' :
        liste_real = []
      else :
         df_real_choisi = df_realisateurs.loc[df_realisateurs['nconst'] == IDactor]['tconst']
         df = pd.merge(df_titres,df_real_choisi, how = 'inner', on = 'tconst')
         df = df.sort_values('notePonderee', ascending = False).head(10-len(liste_finale))
         liste_real = df['titreVF'].tolist()
         liste_finale = liste_finale + liste_real
    return liste_finale

######################## TEST de L'AFFICHAGE

with st.sidebar:
        titre_test = st.text_input("Entrez votre critère de suggestion : ")
        requete_trouvee = 0
if titre_test is not None :
  if titre_test in df_titres['titreVF'].tolist() : # On cherche si c'est un titre de film
    requete_trouvee = 1
    film = id_du_film(titre_test)
    df = df_tri(film)
    films_finaux = suggestions(df, film)

  elif titre_test in df_names['Nom'].tolist(): # On cherche si c'est un acteur ou un réalisateur
    requete_trouvee = 1
    films_finaux = filmograhie(titre_test)      



  if requete_trouvee == 1 :
    col1, col2  = st.columns(2)
    if len(films_finaux) == 0 :
      st.write("Nous n'avons pas trouvé de film")
    with col1: 
        st.write(films_finaux[0])
        if len(films_finaux) >= 3 :
            st.write(films_finaux[2])
        if len(films_finaux) >= 5 :    
            st.write(films_finaux[4])
        if len(films_finaux) >= 7 : 
            st.write(films_finaux[6])
        if len(films_finaux) >= 9 : 
            st.write(films_finaux[8])
    with col2: 
        st.write(films_finaux[1])
        if len(films_finaux) >= 4 : 
            st.write(films_finaux[3])
        if len(films_finaux) >= 6 : 
            st.write(films_finaux[5])
        if len(films_finaux) >= 8 : 
            st.write(films_finaux[7])
        if len(films_finaux) >= 10 : 
            st.write(films_finaux[9])
  else :
    st.write("Nous n'avons pas trouvé de résultat à votre recherche")
     

