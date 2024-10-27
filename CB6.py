import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import messagebox
import speech_recognition as sr
import pyttsx3

class MedicalChatbot:
    def __init__(self, data_path='medical_data.csv'):
        self.data_path = data_path
        self.label_encoder = LabelEncoder()
        self.question_bank = self.load_questions()
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', MultinomialNB())
        ])
        self.load_data()
        self.skip_remaining_questions = False  # To manage skipping questions
        self.engine = pyttsx3.init()  # Text-to-speech engine

    def load_questions(self):
        return [
            "Are you experiencing fever?",
            "Do you feel fatigue or weakness?",
            "Do you have a headache?",
            "Do you have a cough?",
            "Is there any difficulty breathing?",
            "Do you have a runny nose or sore throat?",
            "Do you have any stomach pain?",
            "Have you experienced nausea or vomiting?",
            "Any changes in appetite?",
            "Are you experiencing chest pain?",
            "Do you have shortness of breath?",
            "Any rapid or irregular heartbeat?",
            "Do you have any rash?",
            "Is there itching or swelling?",
            "Have you noticed any skin changes?"
        ]

    def load_data(self):
        try:
            df = pd.read_csv(self.data_path)
            symptoms_cols = [col for col in df.columns if col != 'disease']
            self.symptoms_data = [' '.join([symptom for symptom in row.index if row[symptom] == 1 or row[symptom] == '1'])
                                  for _, row in df[symptoms_cols].iterrows()]
            self.diagnoses = self.label_encoder.fit_transform(df['disease'])
            self.train_model()
        except FileNotFoundError:
            self.create_sample_data()
            self.load_data()

    def create_sample_data(self):
        sample_data = {
            'fever': [1, 0, 1, 0, 1],
            'headache': [1, 0, 1, 0, 0],
            'cough': [1, 1, 0, 0, 0],
            'runny_nose': [1, 1, 0, 0, 0],
            'fatigue': [1, 0, 1, 1, 1],
            'nausea': [0, 0, 1, 0, 0],
            'vomiting': [0, 0, 1, 0, 0],
            'chest_pain': [0, 0, 0, 1, 0],
            'shortness_breath': [0, 1, 0, 1, 0],
            'rash': [0, 0, 0, 0, 1],
            'itching': [0, 0, 0, 0, 1],
            'disease': ['Common Cold', 'Bronchitis', 'Gastroenteritis', 'Respiratory Infection', 'Allergic Reaction']
        }
        pd.DataFrame(sample_data).to_csv(self.data_path, index=False)

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.symptoms_data, self.diagnoses, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def get_user_symptoms_voice(self):
        self.symptoms = []
        self.question_index = 0
        self.skip_remaining_questions = False  # Reset skip flag for new session
        self.ask_question_voice()

    def ask_question_voice(self):
        if self.question_index < len(self.question_bank):
            question = self.question_bank[self.question_index]
            self.speak(question)
            self.show_voice_symptom_prompt(question)
        else:
            self.get_diagnosis_and_recommend()

    def show_voice_symptom_prompt(self, question):
        dialog = tk.Toplevel()
        dialog.title("Voice Symptom Inquiry")
        dialog.geometry("400x300")
        dialog.configure(bg='black')

        label = tk.Label(dialog, text=question, font=("Arial", 14), bg='black', fg='white')
        label.pack(pady=20)

        button_frame = tk.Frame(dialog, bg='black')
        button_frame.pack(pady=10)

        yes_button = tk.Button(button_frame, text="Yes", command=lambda: self.record_symptom_voice(dialog, True), bg="#4CAF50", fg="white", font=("Arial", 12))
        yes_button.pack(side=tk.LEFT, padx=5)

        no_button = tk.Button(button_frame, text="No", command=lambda: self.record_symptom_voice(dialog, False), bg="#ec0000", fg="white", font=("Arial", 12))
        no_button.pack(side=tk.LEFT, padx=5)

        skip_button = tk.Button(button_frame, text="Skip", command=lambda: self.skip_voice_questions(dialog), bg="#FFA500", fg="white", font=("Arial", 12))
        skip_button.pack(side=tk.LEFT, padx=5)

        dialog.wait_window()

    def skip_voice_questions(self, dialog):
        dialog.destroy()
        self.skip_remaining_questions = True  # Set the flag to skip remaining questions
        self.get_diagnosis_and_recommend()  # Directly get diagnosis

    def record_symptom_voice(self, dialog, answer):
        if answer:
            symptom = self.question_bank[self.question_index]
            self.symptoms.append(symptom)
        dialog.destroy()
        self.question_index += 1
        self.ask_question_voice()

    def get_user_symptoms(self):
        self.symptoms = []
        self.question_index = 0
        self.skip_remaining_questions = False  # Reset skip flag for new session
        self.ask_question()

    def ask_question(self):
        if self.question_index < len(self.question_bank):
            question = self.question_bank[self.question_index]
            self.show_symptom_prompt(question)
        else:
            self.get_diagnosis_and_recommend()

    def show_symptom_prompt(self, question):
        dialog = tk.Toplevel()
        dialog.title("Symptom Inquiry")
        dialog.geometry("400x300")
        dialog.configure(bg='black')

        label = tk.Label(dialog, text=question, font=("Arial", 14), bg='black', fg='white')
        label.pack(pady=20)

        button_frame = tk.Frame(dialog, bg='black')
        button_frame.pack(pady=10)

        yes_button = tk.Button(button_frame, text="Yes", command=lambda: self.record_symptom(dialog, True), bg="#4CAF50", fg="white", font=("Arial", 12))
        yes_button.pack(side=tk.LEFT, padx=5)

        no_button = tk.Button(button_frame, text="No", command=lambda: self.record_symptom(dialog, False), bg="#ec0000", fg="white", font=("Arial", 12))
        no_button.pack(side=tk.LEFT, padx=5)

        skip_button = tk.Button(button_frame, text="Skip", command=lambda: self.skip_questions(dialog), bg="#FFA500", fg="white", font=("Arial", 12))
        skip_button.pack(side=tk.LEFT, padx=5)

        dialog.wait_window()

    def skip_questions(self, dialog):
        dialog.destroy()
        self.skip_remaining_questions = True  # Set the flag to skip remaining questions
        self.get_diagnosis_and_recommend()  # Directly get diagnosis

    def record_symptom(self, dialog, answer):
        if answer:
            symptom = self.question_bank[self.question_index]
            self.symptoms.append(symptom)
        dialog.destroy()
        self.question_index += 1
        self.ask_question()

    def get_diagnosis_and_recommend(self):
        symptoms_text = " ".join(self.symptoms)
        prediction, confidence, top_3_diagnoses = self.get_diagnosis(symptoms_text)

        message = f"\nTop 3 possible diagnoses:\n"
        for disease, prob in top_3_diagnoses:
            message += f"- {disease}: {prob:.1f}% confidence\n"
        message += f"\nMost likely diagnosis: {prediction}\n"
        message += f"Confidence: {confidence:.1f}%\n\nRecommendations:\n"
        for rec in self.provide_recommendations(prediction):
            message += f"- {rec}\n"

        messagebox.showinfo("Diagnosis Result", message)

    def get_diagnosis(self, symptoms):
        prediction_idx = self.model.predict([symptoms])[0]
        probabilities = self.model.predict_proba([symptoms])[0]
        confidence = probabilities[prediction_idx] * 100
        prediction = self.label_encoder.classes_[prediction_idx]
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        top_3_diagnoses = [(self.label_encoder.classes_[idx], probabilities[idx] * 100) for idx in top_3_idx]
        return prediction, confidence, top_3_diagnoses

    def provide_recommendations(self, diagnosis):
        recommendations = {
            "Common Cold": [
                "Stay hydrated.",
                "Rest and sleep.",
                "Over-the-counter medications can relieve symptoms."
            ],
            "Bronchitis": [
                "Avoid smoking and secondhand smoke.",
                "Drink plenty of fluids.",
                "Consider a cough suppressant."
            ],
            "Gastroenteritis": [
                "Stay hydrated.",
                "Rest and avoid solid food for a few hours.",
                "Gradually reintroduce bland foods."
            ],
            "Respiratory Infection": [
                "Stay home and rest.",
                "Use a humidifier to ease breathing.",
                "Consult a doctor if symptoms worsen."
            ],
            "Allergic Reaction": [
                "Identify and avoid triggers.",
                "Use antihistamines as needed.",
                "Consult a doctor for severe reactions."
            ]
        }
        return recommendations.get(diagnosis, ["Consult a healthcare professional."])

def main():
    root = tk.Tk()
    root.title("Vital Vision")
    root.geometry("500x400")
    root.configure(bg='black')

    chatbot = MedicalChatbot()

    title_label = tk.Label(root, text="Welcome to Vital Vision", font=("Arial", 20), bg='black', fg='white')
    title_label.pack(pady=20)

    text_button = tk.Button(root, text="Start Text-Based Diagnosis", command=chatbot.get_user_symptoms, bg="#FFA500", fg="white", font=("Arial", 12))
    text_button.pack(pady=10)

    voice_button = tk.Button(root, text="Start Voice-Based Diagnosis", command=chatbot.get_user_symptoms_voice, bg="#4CAF50", fg="white", font=("Arial", 12))
    voice_button.pack(pady=10)

    exit_button = tk.Button(root, text="Exit", command=root.quit, bg="#FF0000", fg="white", font=("Arial", 12))
    exit_button.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    main()
