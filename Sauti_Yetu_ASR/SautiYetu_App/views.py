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
    return render(request, 'patient-sessions.html')


def patients(request):
    return render(request, 'patients.html')


def therapy_session(request):
    return render(request, 'therapy-session.html')


def sessions(request):
    return render(request, 'sessions.html')


def patient_signin(request):
    return render(request, 'patient-signin.html')   


def patient_signup(request):
    return render(request, 'patient-signup.html')


def patient_settings(request):
    return render(request, 'patient-settings.html')


# Therapy Views
def therapist_signup(request):  
    return render(request, 'therapist-signup.html')


def therapist_signin(request):
    return render(request, 'therapist-signin.html')

def therapist_dashboard(request):
    return render(request, 'therapist-dashboard.html')


def therapist_patients(request):
    return render(request, 'therapist-patients.html')


def therapist_sessions(request):
    return render(request, 'therapist-sessions.html')


def therapist_session_details(request):
    return render(request, 'therapist-session-details.html')


def therapist_analytics(request):
    return render(request, 'therapist-analytics.html')



def live_session(request):
    return render(request, 'live-session.html')