import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from deepface import DeepFace
import os

# Function to capture an image using the device's camera
def capture_image():
    cap = cv2.VideoCapture(0)
    result, image = cap.read()
    cap.release()
    if result:
        image_path = 'captured_image.jpg'
        cv2.imwrite(image_path, image)
        entry_image_path.delete(0, tk.END)
        entry_image_path.insert(0, image_path)
        proceed_to_analysis(image_path)
    else:
        messagebox.showerror("Capture Error", "Failed to capture image from the camera.")

# Define the new color scheme
background_color = "#2C2C54"
text_color = "#FFC1C1"
button_color = "#3D8361"
entry_bg_color = "#4D7C8A"
accent_color = "#9B5DE5"

# Define the font styles
title_font = ("Comic Sans MS", 20, "bold")
label_font = ("Comic Sans MS", 12)
button_font = ("Comic Sans MS", 12, "bold")
input_font = ("Comic Sans MS", 12)

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        entry_image_path.delete(0, tk.END)
        entry_image_path.insert(0, file_path)

def analyze_image(image_path):
    analysis = DeepFace.analyze(cv2.imread(image_path), actions=('age', 'gender', 'race'), enforce_detection=False) 
    results = analysis if not isinstance(analysis, list) else analysis[0]

    # Ensure the 'age' key exists in the results
    age = results.get('age', "N/A")

    # Gender result might not be just 'Male' or 'Female', but also include percentages
    # Check the actual value for the gender key in the results
    gender_analysis = results.get('gender', {})
    if isinstance(gender_analysis, dict):
        gender = "Man" if gender_analysis.get('Man') > gender_analysis.get('Woman') else "Woman"
    else:
        # If the gender analysis is not a dictionary, default to the string value
        gender = "Man" if gender_analysis == "Male" else "Woman"

    # Ensure the 'race' key exists and is a dictionary
    races = results.get('race', {})
    race_probabilities = ', '.join([f"{k}: {v:.2f}%" for k, v in races.items()]) if isinstance(races, dict) else "N/A"

    return age, gender, race_probabilities


def proceed_to_analysis(image_path=None):
    image_path = image_path or entry_image_path.get()
    if image_path:
        try:
            if not os.path.exists(image_path):
                raise ValueError("Image file does not exist.")
            age, gender, race_probabilities = analyze_image(image_path)
            display_results(image_path, age, gender, race_probabilities)
        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))
    else:
        messagebox.showerror("Analysis Error", "No image has been selected or captured.")

def display_results(image_path, age, gender, race_probabilities):
    # Hide the home page and show the results page
    home_frame.grid_forget()
    results_frame.grid(row=0, column=0, sticky='nsew')

    # Load and display the image
    load = cv2.imread(image_path)
    cv2.imshow('Uploaded Image', load)

    # Update the labels with the analysis results
    label_name.config(text=f"Name: {entry_name.get()}")
    label_gender.config(text=f"Gender: {gender}")
    label_age.config(text=f"Age: {age}")
    label_race.config(text=f"Races: {race_probabilities}")

def return_home():
    results_frame.grid_forget()
    home_frame.grid(row=0, column=0, sticky='nsew')
    cv2.destroyAllWindows()

# Main window setup
root = tk.Tk()
root.title("Image Analysis Portal")
root.configure(bg=background_color)
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)
root.geometry('600x400')

# Home frame
home_frame = tk.Frame(root, bg=background_color)
home_frame.grid(row=0, column=0, sticky='nsew')

# Results frame
results_frame = tk.Frame(root, bg=background_color)
results_frame.grid(row=0, column=0, sticky='nsew')
results_frame.grid_forget()  # Initially hidden

# Home frame widgets
label_welcome = tk.Label(home_frame, text="Welcome to the Image Analysis Portal", bg=background_color, fg=accent_color, font=title_font)
label_enter_name = tk.Label(home_frame, text="Enter your name:", bg=background_color, fg=text_color, font=label_font)
entry_name = tk.Entry(home_frame, bg=entry_bg_color, fg=text_color, font=input_font)
label_upload = tk.Label(home_frame, text="Please upload your image for analysis:", bg=background_color, fg=text_color, font=label_font)
entry_image_path = tk.Entry(home_frame, bg=entry_bg_color, fg=text_color, font=input_font)
button_upload = tk.Button(home_frame, text="Browse", command=upload_image, bg=button_color, fg=text_color, font=button_font)
button_capture = tk.Button(home_frame, text="Take a Photo", command=capture_image, bg=button_color, fg=text_color, font=button_font)
button_proceed = tk.Button(home_frame, text="Analyze", command=proceed_to_analysis, bg=accent_color, fg=text_color, font=button_font)

# Place home frame widgets
label_welcome.pack(fill='x', pady=10)
label_enter_name.pack(fill='x', pady=10)
entry_name.pack(fill='x', pady=10)
label_upload.pack(fill='x', pady=10)
entry_image_path.pack(fill='x', pady=10)
button_upload.pack(side='left', fill='x', expand=True, pady=10, padx=5)
button_capture.pack(side='left', fill='x', expand=True, pady=10, padx=5)
button_proceed.pack(side='left', fill='x', expand=True, pady=10, padx=5)

# Results frame widgets
label_name = tk.Label(results_frame, bg=background_color, fg=text_color, font=label_font)
label_gender = tk.Label(results_frame, bg=background_color, fg=text_color, font=label_font)
label_age = tk.Label(results_frame, bg=background_color, fg=text_color, font=label_font)
label_race = tk.Label(results_frame, bg=background_color, fg=text_color, font=label_font)
button_return = tk.Button(results_frame, text="Return", command=return_home, bg=button_color, fg=text_color, font=button_font)

# Place results frame widgets
label_name.pack(fill='x', pady=10)
label_gender.pack(fill='x', pady=10)
label_age.pack(fill='x', pady=10)
label_race.pack(fill='x', pady=10)
button_return.pack(fill='x', pady=10)

root.mainloop()
