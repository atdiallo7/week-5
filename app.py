import streamlit as st

from apputil import *

# Load Titanic dataset
df = pd.read_csv('https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv')

st.write(
'''
# Titanic Visualization 1

'''
)
st.write("How does survival rate vary across passenger class, sex, and age group, and which of these three factors seems to matter most?")
# Generate and display the figure
fig1 = visualize_demographic()
st.plotly_chart(fig1, use_container_width=True)

st.write(
'''
# Titanic Visualization 2
'''
)
st.write(
    "Does average ticket fare rise with family size, and does that relationship "
    "(and the spread between the cheapest and priciest tickets) look different "
    "across the three passenger classes?"
)

name_counts = last_names()
st.write(
    f"The most common last name, '{name_counts.index[0]}', appears "
    f"{name_counts.iloc[0]} times, and several other surnames appear 5+ times. "
    "This does **not** fully agree with the family_size table above: `family_size` "
    "is computed per passenger from their own sibling/spouse and parent/child counts, "
    "so two passengers who share a surname (e.g., cousins, in-laws, or unrelated "
    "passengers with the same common name) can inflate the last-name count well "
    "above what any single passenger's own `family_size` would suggest. In other "
    "words, shared surnames capture extended/traveling groups, while `family_size` "
    "only reflects each passenger's immediate family aboard."
)

# Generate and display the figure
fig2 = visualize_families()
st.plotly_chart(fig2, use_container_width=True)

st.write(
'''
# Titanic Visualization Bonus
'''
)
# Generate and display the figure
fig3 = visualize_family_size()
st.plotly_chart(fig3, use_container_width=True)