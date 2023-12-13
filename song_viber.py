# -*- coding: utf-8 -*-
"""
Created on Friday Dec 8 1:18 PM 2023

@author: Sanford Baran
"""

#import libs
import numpy as np
import pandas as pd
import streamlit as st
import json

#Excel Data Files
EXCEL_DATA_File_35 = 'pitchfork_100_2023_gpt35.xlsx'
EXCEL_DATA_File_4 = 'pitchfork_100_2023_gpt4.xlsx'

#define functions 

# Define a function to replace empty strings
def replace_empty_string(s):
    return s if s != '' else 'No Data Available...'

def load_data(file_name):
    df = pd.read_excel(file_name)
    df.fillna('', inplace=True)
    df['Moods'] = df['Moods'].apply(replace_empty_string)
    #print(df.columns)
    return df

def create_moods_df(row):
    moods_json = row.Moods_JSON
    moods_dict = json.loads(moods_json)
    df_moods = pd.DataFrame(moods_dict)
    df_moods.rename(columns={'VAD': 'Valence | Arousal | Dominance'}, inplace=True) 
    #print(df_moods.columns)
    return df_moods

def get_all_moods(df):
    moods_set = set()
    for row in df.itertuples():
        if len(row.Moods) > 0 and row.Moods != 'No Data Available...':
            l = list(row.Moods.strip().split(' '))
            moods_set.update(l)
        
    return sorted(list(moods_set))

def get_moods(df):
    moods_frequencies = {}
    for row in df.itertuples():
        if len(row.Moods) > 0 and row.Moods != 'No Data Available...':
            moods = list(row.Moods.strip().split(' '))
            for mood in moods:
                if mood not in moods_frequencies:
                    moods_frequencies[mood] = 1
                else:
                    moods_frequencies[mood] += 1

    moods = []               
    for key, value in moods_frequencies.items():
        if value > 1:
            moods.append(key)
            
    return ['All Moods'] + sorted(moods) + ['No Data Available...']

def get_genres(df):
    genre_set = set()
    for row in df.itertuples():
        if len(row.Genre) > 0:
            genre_set.add(row.Genre)
        
    return ['All Genres'] + sorted(list(genre_set))

def get_df_filtered(df):
    selected_columns = ['Ranking', 'Title', 'Artist', 'Genre', 'Moods']

    ranking = st.session_state.ranking
    genre = st.session_state.genre
    mood = st.session_state.mood
    
    if ranking == 'Top 25':
        condition1 = df['Ranking'] <= 25
    elif ranking == '26 - 50':
        condition1 = (df['Ranking'] > 25) & (df['Ranking'] <= 50)
    elif ranking == '51 - 75':
        condition1 = (df['Ranking'] > 50) & (df['Ranking'] <= 75) 
    elif ranking == 'Bottom 25':
        condition1 = (df['Ranking'] > 75) & (df['Ranking'] <= 100)
    else:
        condition1 = df['Ranking'] <= 100

    df_filtered = df.loc[condition1, selected_columns]
     
    if mood != 'All Moods':
        condition2 =  df_filtered['Moods'].str.contains(mood, case=False)
        df_filtered = df_filtered.loc[condition2, selected_columns]
        
    if genre != 'All Genres':
        condition3 =  df_filtered['Genre'] == genre
        df_filtered = df_filtered.loc[condition3, selected_columns]

    return df_filtered


def get_rank_and_title_list(df_filtered):
    rank_and_title_list = ["Select a Row by choosing a 'Ranking' -- 'Song Title'"]
    for row in df_filtered.itertuples():
        rank_and_title_list.append(f'{row.Ranking} -- {row.Title}')
    return rank_and_title_list

def is_castable_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def convert_rank_title_choice_to_index(rank_title_choice):
    ranking = rank_title_choice.split(' ')[0]
    print(ranking)
    if is_castable_to_int(ranking):
        return int(ranking) - 1      
    else:
        return -1

def reset_state():
    st.session_state.mood = 'All Moods'
    st.session_state.genre = 'All Genres'
    st.session_state.ranking = 'All Songs'


#load data
df35 = load_data(EXCEL_DATA_File_35)
df4 = load_data(EXCEL_DATA_File_4)

## Initialize Session State Variables
if 'mood' not in st.session_state:
    st.session_state.mood = 'All Moods'
if 'genre' not in st.session_state:
    st.session_state.genre = 'All Genres'
if 'ranking' not in st.session_state:
    st.session_state.ranking = 'All Songs'
if 'llm' not in st.session_state:
    st.session_state.llm = 'GPT-3.5-Turbo'
if 'moods35' not in st.session_state:
    st.session_state.moods35 = get_moods(df35)
if 'moods4' not in st.session_state:
    st.session_state.moods4 = get_moods(df4)
if 'rank_and_title' not in st.session_state:
    st.session_state.rank_and_title = "Select a Row by choosing a 'Ranking' -- 'Song Title'"

# Initialize other variables
if st.session_state.llm == 'GPT-3.5-Turbo':
    df = df35
else:
    df = df4

genres = get_genres(df)

#steamlit gui
st.set_page_config(layout="wide")

## Sidebar Layout
st.sidebar.header('Filters')

#Streamlit 'selectboxes' have this weird behavior whereby if the list of choices of this input widget 
#changes, (which it will sometimes in the case of switching back and forth between LLM == GPT-3.5-Turbo and LLM == GPT4)
#the output of the Selectbox will change-back to an index = 0.  So we want to compute the correct index 'ind' wrt the new list
#of choices, and then specify that within the widget by setting index = ind. This overcomes this 'Mood Selection' widget
#from automatically resetting back to 'Select a Mood' when switching back and forth between LLM == GPT-3.5-Turbo and LLM == GPT4

# Moods selectbox
if st.session_state.llm == 'GPT-3.5-Turbo':
    moods = st.session_state.moods35
else:
    moods = st.session_state.moods4

mood = st.session_state.mood
for index, word in enumerate(moods):
    if word == mood:
        ind = index
        break
    ind = 0

st.sidebar.selectbox('Select a *Mood*', moods, key='mood', index = ind)

# Genre selectbox
st.sidebar.text("")
st.sidebar.selectbox('Select a *Genre*', genres, key='genre')

# Ranking Range selectbox
st.sidebar.text("")
st.sidebar.selectbox('Select a *Ranking* range', ['All Songs', 'Top 25', '26 - 50', '51 - 75', 'Bottom 25'], key='ranking')
st.sidebar.text("")

df_filtered = get_df_filtered(df)

# Number of Songs found after filtering
st.sidebar.write('Found', len(df_filtered), 'songs')
st.sidebar.text("")
st.sidebar.text("")

# Reset Button
st.sidebar.button("Reset Filters", type="primary", on_click=reset_state)

# LLM Radio Buttons
st.sidebar.text("")
st.sidebar.text("")
st.sidebar.text("")
st.sidebar.radio('LLM', ['GPT-3.5-Turbo', 'GPT-4'], key = 'llm')


# Main area Layout

st.markdown("<h1 style='text-align: center; color: black;'>SongViber</h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: gray;'><i>by Sanford Baran</i></h6>", unsafe_allow_html=True)

st.markdown("<h5 style='text-align: center; color: blue;'>PitchFork's Top 100 Songs for 2023</h5>", unsafe_allow_html=True)

st.text("")
st.dataframe(df_filtered, hide_index=True, width=2500)
st.text("")



#Streamlit 'selectboxes' have this weird behavior whereby if the list of choices of this input widget change,
#(which it will sometimes in the case of switching back and forth between LLM == GPT-3.5-Turbo and LLM == GPT4),
#the output of the Selectbox will change-back to an index = 0.  So we want to compute the correct index 'ind' wrt
#the new list of choices, and then specify that within the widget by setting index = ind. This overcomes this
#'Rank and Title Selection'widget from automatically resetting back to "Select a Row by choosing a 'Ranking -- Song Title'"
#when switching back and forth between LLM == GPT-3.5-Turbo and LLM == GPT4

# Row Selector selectbox
rank_and_title_list = get_rank_and_title_list(df_filtered)
rank_and_title = st.session_state.rank_and_title
for index, word in enumerate(rank_and_title_list):
    if word == rank_and_title:
        ind = index
        break
    ind = 0

colA, colB = st.columns([1, 2])
with colA:
    rank_title_choice = st.selectbox('Row Selector', rank_and_title_list, key='rank_and_title', index = ind)

if rank_title_choice != "Select a Row by choosing a 'Ranking' -- 'Song Title'":
    # Display Song Name, Artist, Genre and Position in List
    index = convert_rank_title_choice_to_index(rank_title_choice)
    if index >= 0:
        row = df.iloc[index]

        col1, col2, col3, col4 = st.columns(4)

        st.text("")
        with col1:
            st.markdown("<h4 style='text-align: left; color: black;'>Song Name</h4>", unsafe_allow_html=True)
            st.write(row.Title)

        with col2:
            st.markdown("<h4 style='text-align: left; color: black;'>Artist</h4>", unsafe_allow_html=True)
            st.write(row.Artist)

        with col3:
            st.markdown("<h4 style='text-align: left; color: black;'>Genre</h4>", unsafe_allow_html=True)
            st.write(row.Genre)

        with col4:
            st.markdown("<h4 style='text-align: left; color: black;'>Position in List</h4>", unsafe_allow_html=True)
            st.write(row.Ranking)

        if row.Lyrics == 'Song does not contain lyrics':
            # Display this if song doesn't contain lyrics
            st.text("")
            st.text("")
            st.markdown("<h4 style='text-align: left; color: black;'>Lyrics</h4>", unsafe_allow_html=True)
            st.write(row.Lyrics)

            st.text("")
            st.text("")
            st.markdown("<h4 style='text-align: left; color: black;'>No Data Available</h4>", unsafe_allow_html=True)
        elif row.Moods == 'No Data Available...':
            # .. or this if No Data Available
            st.markdown("<h4 style='text-align: left; color: black;'>No Data Available</h4>", unsafe_allow_html=True)
        else:
            # Lyrics
            st.text("")
            st.text("")
            st.markdown("<h4 style='text-align: left; color: black;'>Lyrics</h4>", unsafe_allow_html=True)
            st.write(row.Lyrics)

            # Meaning
            st.text("")
            st.text("")
            st.markdown("<h4 style='text-align: left; color: black;'>Meaning</h4>", unsafe_allow_html=True)
            st.write(row.Meaning)
            st.caption("(Text generated from OpenAI's *GPT* API)")

            # Moods Summary
            st.text("")
            st.text("")
            st.markdown("<h4 style='text-align: left; color: black;'>Moods Summary</h4>", unsafe_allow_html=True)
            st.write(row.Moods_Description)
            st.caption("(Text generated from OpenAI's *GPT* API)")

            # Emotions Description
            st.text("")
            st.text("")
            st.markdown("<h4 style='text-align: left; color: black;'>Emotions Description</h4>", unsafe_allow_html=True)
            st.write(row.Emotions_Description)
            st.caption("(Text generated from OpenAI's *GPT* API)")
            st.text("")
            st.text("")

            df_moods = create_moods_df(row)

            # List of Moods 
            st.markdown("<h4 style='text-align: left; color: black;'>Moods List</h4>", unsafe_allow_html=True)
            list_of_moods = row.Moods.replace(' ', ', ')
            st.write(list_of_moods)
            st.caption("(Text generated from OpenAI's *GPT* API)")
            st.text("")

            # Moods/VADS Table
            st.markdown("<h4 style='text-align: left; color: black;'>VADs  (Valence, Arousal,Dominance)</h4>", unsafe_allow_html=True)
            st.dataframe(df_moods, hide_index=True)

             # VAD weighted average
            st.markdown("<h5 style='text-align: left; color: black;'>VAD weighted average</h5>", unsafe_allow_html=True)
            st.write(row.VAD_Centroid)
            st.text("")
            st.text("")

            # Pitchfork Comments
            st.markdown("<h4 style='text-align: left; color: black;'>Pitchfork Song Commentary</h4>", unsafe_allow_html=True)
            st.write(row.Pitchfork_Comments)