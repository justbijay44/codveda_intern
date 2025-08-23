import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
import random
from PIL import Image

with open("svm_pipeline.pkl","rb") as f:
    model = pickle.load(f)

data = pd.read_csv("cleaned_iris.csv")

st.set_page_config(page_title="Predict the Species", layout="wide")

st.title("Analysis with species🌺")

fig, ax = plt.subplots(1,2,figsize=(18,6))

counts_sep = data.groupby('sepal_width')['species'].value_counts().unstack().fillna(0)
counts_sep.plot(kind='bar', stacked=True, ax=ax[0], colormap='viridis')

ax[0].set_xlabel("Sepal Width")
ax[0].set_title("Species distribution per Sepal Width")
ax[0].legend(title="Species")

counts_pet = data.groupby('petal_width')['species'].value_counts().unstack().fillna(0)
counts_pet.plot(kind='bar', stacked=True, ax=ax[1], colormap='viridis')
ax[1].set_xlabel("Petal Width")
ax[1].set_title("Species distribution per Petal Width")
ax[1].legend(title="Species")

st.pyplot(fig)

st.sidebar.title("Which Species is it?🌸")

sepal_width_values = np.sort(data['sepal_width'].unique())
petal_width_values = np.sort(data['petal_width'].unique())

sep_width = st.sidebar.selectbox("Sepal Width",[f"{x:.2f}" for x in sepal_width_values])
pet_width = st.sidebar.selectbox("Petal Width", [f"{x: .2f}" for x in petal_width_values])

if st.sidebar.button("Predict Species"):
    user_input = np.array([[float(sep_width), float(pet_width)]])
    pred_class = model.predict(user_input)[0]

    st.sidebar.success(f"The predicted species is {pred_class}")

    img_folder = "images"
    species_image = [f for f in os.listdir(img_folder) if f.startswith(pred_class)]
    img_file = random.choice(species_image)
    img_path = os.path.join(img_folder, img_file)

    img = Image.open(img_path)
    st.sidebar.image(img, caption=f"{pred_class}", width=300)

st.title("Differences among the species")
st.image("images/difference.png")