import pandas as pd
from openai import OpenAI
import numpy as np
from collections import namedtuple
import ast
import pygame
from io import BytesIO
import speech_recognition as sr
import customtkinter as ctk 

client = OpenAI(api_key='YourOpenAIKEY')
dataset = pd.read_csv('C:\\Users\hamza\OneDrive\Desktop\SINESMAP.csv')

def get_embeddings(input):
    response = client.embeddings.create(
        input=input,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def parse_dataset():
    page_number_list = dataset['Page Number'].tolist()
    page_text_list = dataset['Page Text'].tolist()
    page_embeddings_list = dataset['Embeddings'].tolist()
    for idx, element in enumerate(page_embeddings_list):
        page_embeddings_list[idx] = ast.literal_eval(element)
        for id , item in enumerate(page_embeddings_list[idx]):
            page_embeddings_list[idx][id] = float(item)
    page_embeddings_list = np.array(page_embeddings_list)
    top_k = min(3, len(page_text_list))
    return namedtuple('dataset',
        ['page_text_list',
        'page_embeddings_list',
        'page_numbers_list',
        'top_k'])(
            page_text_list,
            page_embeddings_list,
            page_number_list,
            top_k)

def cosine_distance(x,y):
    return np.dot(np.array(x), np.array(y))

def prepare_contexts(dataset):
    contexts = {}
    for page_text, page_number, embedding in zip(
        dataset.page_text_list,
        dataset.page_numbers_list,
        dataset.page_embeddings_list
    ):
        contexts[(page_text, page_number)] = embedding
    return contexts

def order_document_sections_by_query_similarity(query_embedding, context):
    similar = sorted([(cosine_distance(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in context.items()], reverse=True)
    return similar

def get_semantic_suggestions(prompt):
    Dataset = parse_dataset()
    query_embedding = np.array(get_embeddings(prompt), dtype=float)
    relevant_sections = order_document_sections_by_query_similarity(query_embedding, prepare_contexts(dataset=Dataset))
    top_three = relevant_sections[:Dataset.top_k]
    final = []
    for _, (page_text, page_number) in top_three:
        final.append({
            'page_text': page_text,
            'page_number': page_number
        })
    return final

CHAT_COMPLETIONS_MODEL = "gpt-3.5-turbo"

def complete_chat(prompt_obj):
    reply = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt_obj['user']},
            {"role": "system", "content": prompt_obj['system']}
        ],
        model=CHAT_COMPLETIONS_MODEL,
        temperature=0.7
    )
    return reply

import sys
SYSTEM_DEFAULT_PROMPT= "You are a receptionist for SINES a department of NUST University. Your response will be from the context of SINES document. Your job is to answer visitors queries regarding SINES. Context: *insert text* "

# Initialize SpeechRecognition recognizer
recognizer = sr.Recognizer()

# Create a simple GUI
def search_query():
    query = prompt_entry.get("1.0", ctk.END)
    handle_query(query.strip())

def handle_query(user_prompt):
    string = ""
    relevant_pages = get_semantic_suggestions(user_prompt)
    for page in relevant_pages:
        string += f"Page Number: {page['page_number']}\nPage Info: {page['page_text'].strip()}\n"
    updated_system_prompt = SYSTEM_DEFAULT_PROMPT.replace("*insert text*", string)
    prompt_obj = {
        'user': user_prompt,
        'system': updated_system_prompt
    }
    reply = complete_chat(prompt_obj)
    response_text = reply.choices[0].message.content
    response_label.configure(text=response_text)
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=response_text
    )
    # Get the audio data from the response
    audio_data = response.content

    # Initialize Pygame
    pygame.mixer.init()

    # Play the speech
    with BytesIO(audio_data) as stream:
        pygame.mixer.music.load(stream)
        pygame.mixer.music.play()

        # Wait until the speech finishes playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    # Clean up Pygame
    pygame.mixer.quit()

# STT function
def recognize_speech():
    with sr.Microphone() as source:
        print("Speak now...")
        audio = recognizer.listen(source)

    try:
        query = recognizer.recognize_google(audio)
        prompt_entry.delete("1.0", ctk.END)
        prompt_entry.insert("1.0", query.strip())
    except sr.UnknownValueError:
        print("Sorry, I could not understand what you said.")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


#GUI setup
root = ctk.CTk()
root.title("SINES Receptionist")

ctk.set_appearance_mode("dark")

prompt_label = ctk.CTkLabel(root, text="Ask about SINES:", font=("Helvetica", 30, "bold"))
prompt_label.pack(padx=10, pady=10)

input_frame = ctk.CTkFrame(root)
input_frame.pack(side="top",padx=2,pady=2)

query_label = ctk.CTkLabel(input_frame, text="Query:",font=("Helvetica", 14, "bold"))
query_label.grid(row=0,column=0, padx=10, pady=10)
prompt_entry = ctk.CTkTextbox(input_frame, height=50,width=800)
prompt_entry.grid(row=0,column=1, padx=10, pady=10)

stt_button = ctk.CTkButton(input_frame, text="Speak to Search",command=recognize_speech)
stt_button.grid(row=0,column=2, padx=10, pady=10)

search_button = ctk.CTkButton(input_frame, text="Search",command=search_query)
search_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)


response1_label = ctk.CTkLabel(input_frame, text="Response:", font=("Helvetica", 14))
response1_label.grid(row=4, column=0, padx=10, pady=10,)

response_label = ctk.CTkLabel(input_frame, text="", font=("Helvetica", 14))
response_label.grid(row=4, column=1, columnspan=2, sticky="news", padx=10, pady=10,)

root.mainloop()
