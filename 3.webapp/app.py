#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template, Response, session
import pickle
import pandas as pd
import time
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
from datetime import datetime
from fpdf import FPDF


APPOINTMENTS_FILE = "appointments.txt"





import os

import google.generativeai as genai

genai.configure(api_key="AIzaSyALSQg60p7vqNBdn5SHpFKhu0AE8lpe1cE")
# Create the model
# See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
]







# Initialize chat session globally for continuous conversation
chat_session = None




def generate_report(stress):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, txt="Mentally - Your Mental Health Assistant", ln=True, align='C')
    pdf.set_font('Arial', '', 12)
    pdf.cell(200, 10, txt="Your Report", ln=True, align='C') 
    
    f = open("Username/username.txt", "r")
    username = f.read()
    pdf.set_font('Arial', '', 10)
    pdf.cell(100, 10, txt="Name: " + username, align='L')
    date = datetime.today().strftime('%Y-%m-%d')  # Format: YYYY-MM-DD
    pdf.cell(100, 10, txt="Date: "+date, align='R') 
    pdf.ln(10)


    stress = int(float(stress))
    stress = round(stress, 2)
    print(stress)
    pdf.set_font('Arial', '', 12)
    pdf.cell(200, 10, txt="Your final stress percentage: "+ str(stress) + " %", align='L')
    pdf.ln(10)
    
    
    if stress <= 35:
        suggestions = "You're doing great! Keep up your healthy habits. Try practicing mindfulness and continue with your routine."
    elif 35 < stress <= 65:
        suggestions = "Your stress level is moderate. Consider taking short breaks, doing deep breathing exercises, and engaging in relaxing activities."
    else:  # stress > 70
        suggestions = "Your stress level is high. Try meditation, talking to a close friend or therapist, and ensure you're getting enough sleep and exercise."

    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, txt="Suggestions for you: " + suggestions)

    # Save PDF
    pdf.output('Report/report.pdf')


# Load the trained model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Load trained weights
model.load_weights('model.h5')

# Load face cascade for face detection
#facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector = FaceDetector()

# Define emotion categories
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
stress_emotions = {"Angry", "Disgusted", "Fearful", "Sad"}  # Emotions that contribute to stress

# Initialize stress tracking
total_frames = 0
stress_count = 0




#Initialize the flask App
app = Flask(__name__)





#default page of our web-app
@app.route('/')
def landing():
    return render_template('1.landing.html')


@app.route('/loginAd',methods=['POST'])
def loginAd():
    if request.method == 'POST':
        return render_template('2.loginAd.html')


@app.route('/adhome',methods=['POST'])
def adhome():
    if request.method == 'POST':
        lcredentials = [(x) for x in request.form.values()]
        print(lcredentials)
        lusername = lcredentials[0]
        lpassword = lcredentials[1]
        print(type(lusername))


        if lusername=='adm' and lpassword=='123':
            print('match')
            template = '4.Adhome.html'
            try:
                with open(APPOINTMENTS_FILE, "r") as file:
                    appointments = file.read().split("=" * 40 + "\n")  # Split by separators
                    appointments = [a.strip() for a in appointments if a.strip()]  # Remove empty entries
            except FileNotFoundError:
                appointments = []
        elif lusername!='adm' or lpassword!='123':
            print('No')
            template = '3.Adloginfailed.html'
            appointments = []

        return render_template(template,  appointments=appointments)


@app.route('/user',methods=['POST'])
def user():
    if request.method == 'POST':
        return render_template('5.user.html')


@app.route('/signup',methods=['POST'])
def signup():
    if request.method == 'POST':
        return render_template('6.signupusr.html')


@app.route('/signupsuccess',methods=['POST'])
def signupsuccess():
    if request.method == 'POST':
        credentials = [(x) for x in request.form.values()]
        print(credentials)
        username = credentials[0]
        password = credentials[1]
        print(type(username))

        file = open("Username/username.txt", "w")
        a = file.write(username)
        file.close()

        file = open("Password/password.txt", "w")
        a = file.write(password)
        file.close()
        return render_template('7.signupsuccess.html')


@app.route('/login',methods=['POST'])
def login():
    if request.method == 'POST':
        return render_template('8.loginusr.html')


@app.route('/home',methods=['POST'])
def home():
    if request.method == 'POST':
        lcredentials = [(x) for x in request.form.values()]
        print(lcredentials)
        lusername = lcredentials[0]
        lpassword = lcredentials[1]
        print(type(lusername))

        f = open("Username/username.txt", "r")
        username = f.read()
        f = open("Password/password.txt", "r")
        password = f.read()
        print(lusername, username, lpassword, password)

        if username==lusername and password==lpassword:
            print('match')
            template = '10.Usrhome.html'
        elif username!=lusername or password!=lpassword:
            print('No')
            template = '9.loginfailed.html'

        return render_template(template)

@app.route('/home2')
def home2():
    return render_template('10.Usrhome.html')

@app.route('/home3',methods=['POST'])
def home3():
    if request.method == 'POST':
        return render_template('10.Usrhome.html')

@app.route('/qna1')
def qna1():
    return render_template('14.QnA.html')

@app.route('/qna11',methods=['POST'])
def qna11():
    if request.method == 'POST':
        return render_template('14.QnA.html')

@app.route('/suggestions',methods=['POST'])
def suggestions():
    if request.method == 'POST':
        return render_template('19.suggestions.html')


@app.route('/qna2',methods=['POST'])
def qna2():
    if request.method == 'POST':
        return render_template('14.QnA2.html')


@app.route('/landing2',methods=['POST'])
def landing2():
    if request.method == 'POST':
        return render_template('1.landing.html')


@app.route('/instruct',methods=['POST'])
def instruct():
    if request.method == 'POST':
        return render_template('21.instruct.html')

@app.route('/about')
def about():
    return render_template('13.about.html')


@app.route('/detect')
def detect():
    return render_template('15.detect.html')

@app.route('/chat')
def chat():
    return render_template('16.chatbot.html')


@app.route('/consult')
def consult():
    return render_template('17.consult.html') 

@app.route('/consult2',methods=['POST'])
def consult2():
    if request.method == 'POST':
        return render_template('17.consult.html') 



@app.route('/confirm_appointment',methods=['POST'])
def confirm_appointment():
    if request.method == 'POST':
        patient_name = request.form['patient_name']
        gender = request.form['gender']
        phone = request.form['phone']
        appointment_date = request.form['appointment_date']
        doctor = request.form['doctor']
        appointment_time = request.form['appointment_time']
        print(patient_name, gender, phone, appointment_date, appointment_time, doctor)

        with open(APPOINTMENTS_FILE, "a") as file:
            file.write(f"Patient Name: {patient_name}\n")
            file.write(f"Gender: {gender}\n")
            file.write(f"Phone: {phone}\n")
            file.write(f"Appointment Date: {appointment_date}\n")
            file.write(f"Doctor: {doctor}\n")
            file.write(f"Appointment Time: {appointment_time}\n")
            file.write("="*40 + "\n")  # Separator for readability

        return render_template('20.done.html')


def generate_frames():
    # Initialize stress tracking
    total_frames = 0
    stress_count = 0


    global fps
    camera = cv2.VideoCapture(0)
    frame_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(frame_width)


    while True:
        success, frame = camera.read()
        if not success:
            break
        else:

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img, bboxs = detector.findFaces(frame)
            faces = []
            for i in range(len(bboxs)):
                faces.append(bboxs[i]["bbox"])
            

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

                prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                emotion = emotion_dict[maxindex]
                cv2.putText(frame, emotion, (x + 75, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


              

                total_frames += 1
                if emotion in stress_emotions:
                    stress_count += 1


         
            stress_percentage = (stress_count / total_frames) * 100 if total_frames > 0 else 0
            print(stress_percentage)
            cv2.putText(frame, f"Stress Level: {stress_percentage:.2f}%", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


            # Create a simple graphical representation (progress bar) for stress
            bar_x, bar_y, bar_w, bar_h = 20, 100, 300, 30  # Set position and size of bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)  # Outline of bar
            fill_w = int(bar_w * (stress_percentage / 100))  # Calculate filled portion based on stress percentage
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 0, 255), -1)  # Fill bar with color
            


            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    print('YESSSSSSSSSSSSSSSSSS')
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

















def generate_frames2():
    # Initialize stress tracking
    total_frames = 0
    stress_count = 0


    global fps
    camera = cv2.VideoCapture(0)
    frame_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(frame_width)



    start_time = time.time()
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img, bboxs = detector.findFaces(frame)
            faces = []
            for i in range(len(bboxs)):
                faces.append(bboxs[i]["bbox"])
            

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

                prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                emotion = emotion_dict[maxindex]
                cv2.putText(frame, emotion, (x + 75, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


              

                total_frames += 1
                if emotion in stress_emotions:
                    stress_count += 1


         
            stress_percentage = (stress_count / total_frames) * 100 if total_frames > 0 else 0
            print('Final Stress is', stress_percentage)
            #file = open("stress_level.txt", "w")
            #a = file.write(stress_percentage)
            #file.close()

            #session['stress_percentage'] = stress_percentage
            cv2.putText(frame, f"Stress Level: {stress_percentage:.2f}%", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            

            # Create a simple graphical representation (progress bar) for stress
            bar_x, bar_y, bar_w, bar_h = 20, 100, 300, 30  # Set position and size of bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)  # Outline of bar
            fill_w = int(bar_w * (stress_percentage / 100))  # Calculate filled portion based on stress percentage
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 0, 255), -1)  # Fill bar with color
            


            # Save final stress percentage to a file
            with open("stress_percentage.txt", "w") as file:
                file.write(str(stress_percentage))
            #generate_report(stress_percentage)


            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            


@app.route('/video_feed2')
def video_feed2():
    print('YESSSSSSSSSSSSSSSSSS')
    return Response(generate_frames2(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/test',methods=['POST'])
def test():
    if request.method == 'POST':
        return render_template('23.test.html')


@app.route('/final_page')
def final_page():
    f = open("stress_percentage.txt", "r")
    percent = f.read()
    print(type(percent))
    generate_report(percent)
    percent = int(float(percent))
    percent = round(percent, 2)
    if percent <= 35:
        suggestions = "You're doing great! Keep up your healthy habits. Try practicing mindfulness and continue with your routine."
    elif 35 < percent <= 65:
        suggestions = "Your stress level is moderate. Consider taking short breaks, doing deep breathing exercises, and engaging in relaxing activities."
    else:  # stress > 70
        suggestions = "Your stress level is high. Try meditation, talking to a close friend or therapist, and ensure you're getting enough sleep and exercise."
    return render_template('22.final.html', percent=percent, suggestions=suggestions)



# Route to handle chat messages
@app.route('/ask', methods=['POST'])
def ask_question():
    model = genai.GenerativeModel(
  model_name="gemini-1.5-flash-latest",
  safety_settings=safety_settings,
  generation_config=generation_config,
)
    global chat_session  # Use a global variable to maintain the session across requests

    user_message = request.form.get('question')

    # Initialize the chat session once for a continuous conversation
    if chat_session is None:
        chat_session = model.start_chat()

    try:
        # Send user message and get a response
        response = chat_session.send_message('Act as a mental health assistant to reply to user messages and limit your answers simple and in 80 words maximum' + user_message)
        bot_reply = response.text
    except Exception as e:
        bot_reply = "I'm sorry, I couldn't get an answer for that question."

    return jsonify({'answer': bot_reply})


if __name__ == "__main__":
    app.run(debug=True)
