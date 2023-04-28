import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
import random
import os
import json

st.title('Air Cast-  Relaible air quality prediction in your area')
st.write('Enter your zipcode')
# particles = ["NO","PM2.5","PM","OZONE"]
# Define the range of valid zip codes
min_zip = 00000
max_zip = 99999
# particles = ["NO","PM2.5","PM","OZONE"]
# Get the zip code input from the user
zip_input = st.number_input('Enter a zip code', min_value=min_zip, max_value=max_zip)
st.button('Search')

# Check if the zip code is within the valid range
if min_zip <= zip_input <= max_zip:
    st.write('Valid zip code entered:', zip_input)
    
else:
    st.write('Invalid zip code entered:', zip_input)

st.write("The elements for the zipcode you entered")

##----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# Create two columns with the same width
left_column, right_column = st.columns(2)
# Add content to the left column
with left_column:
    st.header('API calls remaining')
    st.write("Number")

# Add content to the right column
with right_column:
    st.header('API calls done')
    st.write("Number ")

##-------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Read elements from file or database
db_elements = ["NO", "PM10", "PM2.5", "OZONE"]
st.write(db_elements)
val = ["NO", "PM2.5","OZONE"]
elements = []
for el in db_elements:
    if el in val:
        elements.append(el)

# check for elements in db
db_elements = []
for v in val:
    # Code to check for element in db goes here
    # Assuming it returns True if element is found
    if True:
        db_elements.append(v)

    # Generating  plots for each existing element in elements list
for el in elements:
    val_dict = {}
    if el == "NO" or el == "OZONE":
        val_dict[el] = random.randint(0, 100)
        fig = go.Figure([go.Bar(x=[el], y=[val_dict[el]])])
    else:
        val_dict[el] = [random.randint(0, 100) for _ in range(10)]
        fig = go.Figure([go.Scatter(x=list(range(10)), y=val_dict[el], mode='lines')])
    fig.update_layout(title="Random Values for " + el, xaxis_title=el, yaxis_title="Value")
    st.plotly_chart(fig)

# Generate combined bar chart for all elements in db_elements
if len(db_elements) > 1:
    combined_dict = {v: random.randint(0, 100) for v in db_elements}
    fig = go.Figure([go.Bar(x=list(combined_dict.keys()), y=list(combined_dict.values()))])
    fig.update_layout(title="Random Values for All Elements", xaxis_title="Element", yaxis_title="Value")
    st.plotly_chart(fig)



    fig = go.Figure([go.Scatter(x=list(combined_dict.keys()), y=list(combined_dict.values()), mode="markers")])
    fig.update_layout(title="Random Values for All Elements", xaxis_title="Element", yaxis_title="Value")
    st.plotly_chart(fig)


#--------------------------------------------------------------------------------------------------------------------------------------
#trying to display the risk factors for each particle and measuring the values based
# Define the threshold values for each particle
thresholds = {"NO": 25, "PM2.5": 12, "PM": 25, "OZONE": 60}

# Define the particles to plot
particles = ["NO", "PM2.5", "PM", "OZONE"]

# Create a dictionary to hold the values for each particle
values = {}
for particle in particles:
    values[particle] = random.randint(0, 100)

# Creating a bar chart for each particle, coloring the bars based on whether they are above or below the threshold
# Create a list of colors based on the thresholds
colors = ["green" if values[particle] <= thresholds[particle] else "red" for particle in particles]
# Create a list of Bar objects for each particle
bars = [go.Bar(x=[particle], y=[values[particle]], marker=dict(color=color)) for particle, color in zip(particles, colors)]
# Create a Figure object with all the bars
fig = go.Figure(data=bars)
# Update the layout of the figure
fig.update_layout(title="Risk range for all particles", xaxis_title="Particle", yaxis_title="Value")
# Display the figure in Streamlit
st.plotly_chart(fig)












#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Recommendations based on ChatGPT


# Set the AQI value and parameter
aqi = 35
parameter = "ozone"


#setting the API KEY here
CHATGPT_API_KEY = "mention the api key here"

# Get the response from the ChatGPT API
url = "https://api.openai.com/v1/engines/davinci/completions"

headers = {
  "Authorization": "Bearer YOUR_API_KEY",
  "Content-Type": "application/json",
}

data = {
  "prompt": "I am an air quality expert. Answer the below in less than 50 words as an air quality expert: what are effects of AQI {} {} what range is it in and what are the effects on health?".format(aqi, parameter),
  "temperature": 0.7,
  "max_tokens": 50,
  "presence_penalty": 0.2,
  "stop_token": ".",
}

response = requests.post(url, headers=headers, data=json.dumps(data))

# Check the response status code
if response.status_code == 200:
  # Get the response data
  response_data = json.loads(response.content)

  # Get the completion
  gpt_response = response_data["choices"][0]["text"]

  # Print the completion
  st.write("Chat GPT completion")
  print(gpt_response)

else:
  if response.status_code == 404:
    print("Error retrieving the response: {}".format(response.status_code))

