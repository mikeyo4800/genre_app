import joblib
import streamlit as st
from preprocessing import TextPreprocessor
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# 1 . button to load in pdf 

st.title("Screenplay Genre Classifier")
st.subheader("In this application, we use a classifier chain with a base estimator to classifiy the genre of our screenplay.")
st.write("For source code/github repo: [GitHub](https://github.com/mikeyo4800/Screenplay_Genre_Classifier)")

uploaded_file = st.file_uploader("Upload a file (txt file only for now)")


lst = []

if uploaded_file:
     
     for line in uploaded_file:
          lst.append(line.decode())

     data = " ".join(lst)

# 2. convert api from pdf to text

# 3. Text preprocessing (lemmatizing and removing stop words)
     
     stop_words = []

     with open('stop_list.txt', 'r') as fp:
          for line in fp:
               x = line[:-1]

        # add current item to the list
               stop_words.append(x)
     
     with open('names.txt', 'r') as f:
          for line in f:
               
               y = line[:-1]

        # add current item to the list
               stop_words.append(y.lower())

     tp = TextPreprocessor(activator_type='wnl', lem_or_stem='lem')
     
     data_lemmatized = tp.lem_process_doc(data)
     
     data_cleaned = " ".join([text for text in data_lemmatized.split() if text not in stop_words])

     

# 4. Word Cloud?
     plt.figure(figsize=(40,25))
     cloud = WordCloud(
                              background_color='black',
                              collocations=True,
                              width=2500,
                              height=1800
                              ).generate(data_cleaned)
     plt.axis('off')
     plt.title('Word Cloud of Script',fontsize=40)
     plt.imshow(cloud)
     st.set_option('deprecation.showPyplotGlobalUse', False)
     st.pyplot()


# 5. model predictions
     with open("model_genre.pkl", "rb") as f:
          model = joblib.load(f)

     predictions = model.predict([data_cleaned])

     genre_names = []          
     genre_dct = {0 : 'Crime', 1:'Romance', 2: 'Animation', 3: 'SciFi', 4: 'Fantasy', 5: 'History', 6: 'Action', 7: 'Drama', 8:'War', 9: 'Thriller', 10: 'Mystery', 11: 'Documentary', 12: 'Horror', 13: 'Family', 14: 'Adventure', 15: 'Music', 16: 'Comedy', 17: 'Western'}

     for i,x in enumerate(predictions[0]):
          if x == 1:
               genre_names.append(genre_dct[i])

     st.write(genre_names)