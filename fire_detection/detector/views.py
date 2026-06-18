from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.decorators import login_required
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Model Config
IMG_SIZE = 128
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = tf.keras.models.load_model(os.path.join(BASE_DIR, "fire_detection_smcnn.keras"))
le = LabelEncoder()
le.fit(np.load(os.path.join(BASE_DIR, "labels.npy")))

def home(request):
    return render(request, 'home.html')


def signup_view(request):
    if request.method == "POST":
        u = request.POST.get('username')
        p = request.POST.get('password')
        cp = request.POST.get('confirm_password')  
        if p == cp:
            User.objects.create_user(username=u, password=p)
            return redirect('login') 
    return render(request, 'signup.html')

def login_view(request):
    if request.method == "POST":
        u = request.POST.get('username')
        p = request.POST.get('password')
        user = authenticate(username=u, password=p)
        if user:
            login(request, user)
            return redirect('index')
    return render(request, 'login.html')
@login_required(login_url='login')
def index(request):
    context = {}
    if request.method == "POST" and request.FILES.get("image"):
        image = request.FILES["image"]
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        img = cv2.imread(fs.path(filename))
        if img is not None:
            img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE)) / 255.0
            pred = model.predict(np.expand_dims(img, axis=0))[0][0]
            label = le.inverse_transform([1 if pred > 0.5 else 0])[0]
            context = {"image_url": fs.url(filename), "label": label}
    return render(request, 'index.html', context)