import streamlit as st
from backend import StableDiffusionBackend
from therapist_agent import TherapistAgent

st.title("AI Therapist")

# Load therapist agent with cached resource
@st.cache_resource
def load_agent():
    sd_backend = StableDiffusionBackend(model_path="/content/dreambooth-leo")
    return TherapistAgent(sd_backend)

agent = load_agent()

# User history input
st.sidebar.header("User Information")
name = st.sidebar.text_input("Your Name", value="Milan Varghese")
age = st.sidebar.number_input("Your Age", min_value=1, max_value=100, value=25)
occupation = st.sidebar.text_input("Occupation", value="Master's Student in AI at Drexel University")
pet_keyword = st.sidebar.text_input("Pet Training Keyword", value="<bruno>")

if "history" not in st.session_state:
    st.session_state["history"] = []

user_input = st.text_input("How are you feeling today?")

if st.button("Chat"):
    with st.spinner("Therapist is responding..."):
        # Combine user history into context string (concise for therapist, detailed for user)
        user_context = (
            f"The user's name is {name}, a {age}-year-old {occupation}. "
            f"The user's pet has been trained with the keyword {pet_keyword}."
        )

        # Therapist interaction (text response)
        therapist_response = agent.chat(user_context + " " + user_input)

    st.session_state["history"].append({"user": user_input, "therapist": therapist_response})

for exchange in reversed(st.session_state["history"]):
    st.write(f"**You:** {exchange['user']}")
    st.write(f"**Therapist:** {exchange['therapist']['response_text']}")

    if exchange['therapist']['generated_image']:
        st.image(exchange['therapist']['generated_image'], caption=exchange['therapist']['image_prompt'])
