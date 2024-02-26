"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import string  
import re  
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorizer
news_vectorizer = open("resources/vectorizer.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home","Prediction","About us"]
	selection = st.sidebar.selectbox("Navigation Pane", options)

	# Building out the "Information" page #Changed to HOME
	if selection == "Home":  #changed to HOME
		st.info("You can describe the whole project in here")
		
		st.image("resources/Taylor1.webp",width=450) #added this
		# You can read a markdown file from supporting resources folder
		st.markdown("Above we have the one and only Taylor Swift")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
		
		st.subheader("I can make more subheadings")
		st.markdown("With st.markdown I can then write things that go below that heading")

		st.subheader("Introducing my cats")
		st.image("resources/Oliver.png",width=450)
		st.markdown("This is Oliver, one of 15 cats")


	# Building out the predication page
	if selection == "Prediction":
		st.info("Here I can write about the models I have that you can choose from and stuff")
		
		#ADDING SIDEBAR!!!
		select = st.sidebar.selectbox('Choose Machine Learning Model ⬇️',['Logistic Regression','Support Vector Classfier (SVC)', 'Support Vector Machine (SVM)'])
		
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			if select == 'Logistic Regression':
				predictor = joblib.load(open(os.path.join("resources/lg_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
			if select == 'Support Vector Classifier (SVC)':
				predictor = joblib.load(open(os.path.join("resources/svc_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
			if select == 'Support Vector Machine (SVM)':
				predictor = joblib.load(open(os.path.join("resources/svm_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)



			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
