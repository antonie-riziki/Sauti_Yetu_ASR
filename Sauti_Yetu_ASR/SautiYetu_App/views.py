from django.shortcuts import render

# Create your views here.
def home(request):
    return render(request, 'index.html')


def community(request):
    return render(request, 'community.html')


def nearby(request):
    return render(request, 'nearby.html')   


def patient_dashboard(request):
    return render(request, 'patient-dashboard.html')


def patient_sessions(request):
    return render(request, 'patient-session.html')


def patients(request):
    return render(request, 'patients.html')


def therapy_session(request):
    return render(request, 'therapy-session.html')


def sessions(request):
    return render(request, 'sessions.html')


def signin(request):
    return render(request, 'signin.html')   


def signup(request):
    return render(request, 'signup.html')


# def therapist_dashboard(request):
#     return render(request, 'therapist-dashboard.html')


# def therapist_session(request):
#     return render(request, 'therapist-session.html')


# def therapist_sessions(request):
#     return render(request, 'therapist-sessions.html')

