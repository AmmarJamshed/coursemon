#!/usr/bin/env python
# coding: utf-8

# In[3]:

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset: Courses with fields and required skills
course_data = [
    {"Course": "Data Science Bootcamp", "Field": "Data Science", "Skills": "Python, Machine Learning, Statistics, SQL"},
    {"Course": "AI for Beginners", "Field": "Artificial Intelligence", "Skills": "Deep Learning, Python, Neural Networks"},
    {"Course": "Business Analytics Masterclass", "Field": "Business Analytics", "Skills": "Excel, SQL, Data Visualization"},
    {"Course": "Web Development Pro", "Field": "Web Development", "Skills": "HTML, CSS, JavaScript, React"},
    {"Course": "Cybersecurity Essentials", "Field": "Cybersecurity", "Skills": "Networking, Ethical Hacking, Cryptography"},
    {"Course": "Cloud Computing Fundamentals", "Field": "Cloud Computing", "Skills": "AWS, Azure, DevOps"},
]

# Convert to DataFrame
df = pd.DataFrame(course_data)

# Streamlit UI
st.title("Course Recommendation Engine 🎓")
st.write("Get the right courses based on your skills and target career field.")

# User Inputs
experience = st.selectbox("Select your experience level:", ["Beginner", "Intermediate", "Advanced"])
skills = st.text_input("Enter your skills (comma-separated):", "Python, SQL")
target_field = st.selectbox("Select your target field:", df["Field"].unique())

# Function to recommend courses
def recommend_courses(user_experience, user_skills, target_field):
    # Create a new user profile
    user_profile = f"{target_field}, {user_experience}, {user_skills}"

    # Combine field and skills to form a text corpus
    df["Combined_Text"] = df["Field"] + ", " + df["Skills"]

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["Combined_Text"])

    # Transform the user profile into the same vector space
    user_vector = vectorizer.transform([user_profile])

    # Calculate cosine similarity
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Get top recommended courses
    df["Similarity"] = similarities
    recommended_courses = df.sort_values(by="Similarity", ascending=False).head(3)

    return recommended_courses[["Course", "Field", "Skills"]]

# Display recommendations on button click
if st.button("Get Recommendations"):
    recommended = recommend_courses(experience, skills, target_field)
    st.subheader("Recommended Courses for You:")
    for i, row in recommended.iterrows():
        st.write(f"✅ **{row['Course']}** ({row['Field']})")
        st.write(f"🔹 Required Skills: {row['Skills']}")


# In[ ]:




