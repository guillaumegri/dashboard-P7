import streamlit as st
import pickle
import joblib
# from sentence_transformers import SentenceTransformer
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time

def load_from_chunks(output_prefix, num_chunks):
    data = b""
    for i in range(num_chunks):
        with open(f"{output_prefix}_part{i}.pkl", 'rb') as chunk_file:
            data += chunk_file.read()
    return pickle.loads(data)
# # Supposons que vous avez un DataFrame avec les colonnes spécifiées
# data = {
#     "Title_text": ["Titre 1", "Titre 2", "Titre 3"],
#     "Body_text": ["Question 1", "Question 2", "Question 3"],
#     "CreationDate": ["2024-01-01", "2024-01-02", "2024-01-03"],
#     "Tags": [["tag1", "tag2"], ["tag1", "tag3"], ["tag2", "tag4"]],
#     "processed_Title_text": ["Titre1", "Titre2", "Titre3"],
#     "processed_Body_text": ["Question1", "Question2", "Question3"],
# }

# df = pd.DataFrame(data)

# with open('df_preprocessed.pkl', 'rb') as file:
#     df = pickle.load(file)
start_time = time.time()
df = load_from_chunks('df_preprocessed_chunk', num_chunks=5)

loading_time = time.time() - start_time
# Calcul des statistiques
num_questions = df.shape[0]
tags_flattened = [tag for sublist in df['processed_Tags'] for tag in sublist]
tags_count = pd.Series(tags_flattened).value_counts()

# Obtenir la fréquence des tags individuels
from collections import Counter
all_tags = [tag for tags in df['processed_Tags'] for tag in tags]
tag_frequencies = Counter(all_tags)

# Attribuer le tag dominant (le plus fréquent dans l'ensemble) à chaque question
df['dominant_tag'] = df['processed_Tags'].apply(lambda x: max(x, key=lambda tag: tag_frequencies[tag]))

# Échantillon stratifié selon le tag dominant
echantillon = df.groupby('dominant_tag').apply(lambda x: x.sample(frac=0.05, random_state=42)).reset_index(drop=True)

print(len(set(tags_flattened)))
count_time = time.time() - start_time - loading_time

text = ' '.join([' '.join(row.processed_Title_text) + ' ' +  ' '.join(row.processed_Body_text) for row in echantillon.itertuples()])
# print(text)
print(loading_time, count_time)

# Configuration du layout
st.set_page_config(layout="wide")

# Colonne de gauche
with st.sidebar:
    st.header("Description des Données")
    st.write(f"**Nombre de questions :** {num_questions}")

    # Affichage des titres, corps et tags
    # for index, row in df.iterrows():
    #     st.subheader(f"Titre : {row['Title_text']}")
    #     st.write(f"Question : {row['Body_text']}")
    #     st.write(f"Tags : {', '.join(row['Tags'])}")
    #     st.write(f"Date de Création : {row['CreationDate']}")
    #     st.markdown("---")  # Séparateur

# Colonne de droite
st.title("Dashboard des Questions de StackOverflow")

# Graphique des fréquences de tags
fig_tags = px.bar(tags_count, x=tags_count[0:20].index, y=tags_count[0:20].values, labels={'x': 'processed_Tags', 'y': 'Fréquence'},
                  title="Fréquence des Tags")
fig_tags.update_layout(
    title={'text': "Fréquence des Tags", 'font_size': 24},  # Taille du titre
    xaxis_title={'text': 'processed_Tags', 'font': {'size': 20}},  # Taille du label de l'axe X
    yaxis_title={'text': 'Fréquence', 'font': {'size': 20}},  # Taille du label de l'axe Y
    xaxis={'tickfont': {'size': 18}},  # Taille des ticks de l'axe X
    yaxis={'tickfont': {'size': 18}}   # Taille des ticks de l'axe Y
)
st.plotly_chart(fig_tags)

# # WordCloud des mots clés
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# st.pyplot(plt)

# Liste de tous les tags uniques
all_tags = sorted(set(tag for tags in df['processed_Tags'] for tag in tags))

# Interface utilisateur : sélection de tags
selected_tags = st.multiselect("Choisissez un ou plusieurs tags", options=all_tags)

# Filtrer les données en fonction des tags sélectionnés
if selected_tags:
    filtered_data = df[df['processed_Tags'].apply(lambda tags: any(tag in tags for tag in selected_tags))]
    # Concaténer les textes filtrés
    text = ' '.join([' '.join(row.processed_Title_text) + ' ' +  ' '.join(row.processed_Body_text) for row in filtered_data.itertuples()])
    
    # Générer le WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    # Afficher le WordCloud
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)
else:
    st.write("Sélectionnez un tag pour voir le nuage de mots correspondant.")

# Histogramme des longueurs de questions
df['Longueur de titre'] = df['Title_text'].str.len()
fig_length = px.histogram(df, x='Longueur de titre', title="Distribution de la Longueur des Titres (en caractères)")
fig_length.update_layout(
    title={'text': "Fréquence des Tags", 'font_size': 24},  # Taille du titre
    xaxis_title={'text': 'processed_Tags', 'font': {'size': 20}},  # Taille du label de l'axe X
    yaxis_title={'text': 'Fréquence', 'font': {'size': 20}},  # Taille du label de l'axe Y
    xaxis={'tickfont': {'size': 18}},  # Taille des ticks de l'axe X
    yaxis={'tickfont': {'size': 18}}   # Taille des ticks de l'axe Y
)
st.plotly_chart(fig_length)
# MODEL_NAME = "avsolatorio/GIST-small-Embedding-v0"

# # Titre de l'application
# st.title("Interface pour Consommer une API")

# # Entrée utilisateur pour l'API (par exemple, un paramètre)
# titre = st.text_input("Titre", "")
# question = st.text_input("Question", "")

# # URL de l'API
# # api_url = "https://apitagspredict-dvhyeehxa8c3avah.germanywestcentral-01.azurewebsites.net"
# # # api_url = 'http://localhost:8000'

# # Bouton pour interroger l'API
# if st.button("Envoyer la requête"):
#     route = "/predict"
    # if question:

        # data = {
        #     "text": [titre + " " + question]
        # }

        # embedding_model = SentenceTransformer(
        #     MODEL_NAME
        # )
        
        # # X_batches = [documents[i:i + batch_size] 
        # #             for i in range(0, len(documents), batch_size)]
        
        # X = embedding_model.encode(data["text"])
        
        # with open('model.pkl', 'rb') as file_model:
        #     model = pickle.load(file_model)

        # with open('mlb.pkl', 'rb') as mlb_file:
        #     mlb = pickle.load(mlb_file)

        # predictions = model.predict(X)

        # tags = mlb.inverse_transform(predictions)    
        # tags = [tag for sublist in tags for tag in sublist]

        # st.write(f"Tags : {', '.join(tags)}")
    # else:
    #     st.warning("Veuillez entrer un paramètre.")
