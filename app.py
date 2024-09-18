import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

# Google API configuration
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load your landmark prediction model
model = tf.keras.models.load_model('my_combined_model.h5')

# Function to load and prepare the image
def load_and_prep_image(image, img_shape=300):
    img = tf.convert_to_tensor(image)
    img = tf.image.decode_image(tf.io.encode_jpeg(img), channels=3)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = img / 255.0
    return img

# Function to generate content using Gemini
def generate_gemini_content(pred_class, prompt):
    try:
        model = genai.GenerativeModel("gemini-pro")
        query = f"{prompt}: {pred_class}"
        response = model.generate_content(query)
        return response.text
    except Exception as e:
        return f"Error generating content: {str(e)}"

# Function to predict the class of a landmark
def pred_and_plot(model, image, class_names):
    img = load_and_prep_image(image)
    pred = model.predict(tf.expand_dims(img, axis=0))
    
    if len(pred[0]) > 1:  # Check for multi-class
        pred_class = class_names[pred.argmax()]
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]
    
    return pred_class

# Set the title of the app
# Add an icon and title
st.set_page_config(page_title="wanderOUR", page_icon="🌍")

# Sidebar with logo and navigation
st.sidebar.image("https://img.icons8.com/?size=100&id=hOIUeiHFcOm1&format=png&color=000000", width=100)  
st.sidebar.title("Menu")
pages = ["Landmark Predictor", "Planner","FarePrice"]
page = st.sidebar.selectbox("Navigate", pages)

# Main Heading with an Icon
st.title("wanderOUR")

# List of class names for prediction
class_names = ['Ajanta Caves','Charar-E- Sharif', 'Chhota_Imambara',
               'Ellora Caves', 'Fatehpur Sikri', 'Gateway of India',
               'Humayun_s Tomb', 'India gate pics', 'Khajuraho',
               'Sun Temple Konark', 'alai_darwaza', 'alai_minar',
               'basilica_of_bom_jesus', 'charminar', 'golden temple',
               'hawa mahal pics', 'iron_pillar', 'jamali_kamali_tomb',
               'lotus_temple', 'mysore_palace', 'qutub_minar', 'tajmahal',
               'tanjavur temple', 'victoria memorial']

# Landmark Predictor page
if page == "Landmark Predictor":
    st.header("Landmark image generator using AI")

    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    prompt = "Provide a comprehensive overview of this landmark, including its historical significance, cultural importance, and key features"

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        pred_class = pred_and_plot(model, image, class_names)
        st.subheader(f"Prediction: {pred_class}")
        gemini_content = generate_gemini_content(pred_class, prompt)
        if st.button("Generate_info"):
            st.write(f"Generated Information: {gemini_content}")
        if st.button("Explore"):
            prompt="Give a list of undiscovered or underrated tourist landmark which holds the same level of heritage culture as the given location. Make sure the underrated places should be withing 100km of the given location"
            gemini_content=generate_gemini_content(prompt,pred_class)
            st.write(f"{gemini_content}")

# Planner page
elif page == "Planner":
    st.header("Itinerary Planner")

    # Input for start date, end date, and location
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    location = st.text_input("Enter Location")
    no_days = end_date - start_date
    trip = location+"Number of days =" + str(no_days.days)
    prompt = "Give me a proper itinerary for a vacation with my family for the given locations for the given number of days. Keep in check with the real-time traffic, weather condition to visit the location and also proper break sessions for lunch/dinner/tea, etc.And provide me with the actual complete google maps url for those places. The location and number of days are as given below"

    if st.button("Plan"):
        if start_date and end_date and location:
            st.write(f"Planning trip to {location} from {start_date} to {end_date}")
            gemini_content = generate_gemini_content(trip, prompt)
            st.write("Best plan for your trip ❤️")
            st.write(f"{gemini_content}")
        else:
            st.write("Please enter all details for the planner.")
elif page == "FarePrice":
    st.header("Travel Cost Estimator")

    # Input for start location, end location, and transport mode
    start_location = st.text_input("Starting Location")
    end_location = st.text_input("Destination Location")
    transport_mode = st.selectbox("Select Mode of Transport", ["Any","Bus", "Train", "Taxi"])

    if st.button("Calculate Fare"):
        if start_location and end_location:
            prompt="Refer to other travelling angencies and find the best agency to travel from the location to the location , using this mode of transport. also give the source for your estimation , and give the fares as well in a tabulated format is appreciated." 
            locs=start_location+end_location
            trip=locs+str(transport_mode)
            st.write(f"Estimated fare from {start_location} to {end_location} by {transport_mode}: ")
            gemini_content=generate_gemini_content(prompt,trip)
            st.write(f"{gemini_content}")
        else:
            st.write("Please enter both starting and destination locations.")
