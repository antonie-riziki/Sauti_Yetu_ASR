from django.urls import path, include
from . import views


urlpatterns = [    
    path('', views.home, name='home'),

    # Patient URLs
    path('community/', views.community, name='community'),
    path('nearby/', views.nearby, name='nearby'),
    path('patient-dashboard/', views.patient_dashboard, name='patient_dashboard'),
    path('patient-sessions/', views.patient_sessions, name='patient_sessions'),
    path('patients/', views.patients, name='patients'),
    path('sessions/', views.sessions, name='session'),
    path('patient-signin/', views.patient_signin, name='patient-signin'),
    path('patient-signup/', views.patient_signup, name='patient-signup'),
    path('patient-settings/', views.patient_settings, name='patient-settings'),

    # Therapy URLs
    path('therapist-signup/', views.therapist_signup, name='therapist_signup'),
    path('therapist-signin/', views.therapist_signin, name='therapist_signin'),
    path('therapist-dashboard/', views.therapist_dashboard, name='therapist_dashboard'),
    path('therapist-patients/', views.therapist_patients, name='therapist_patients'),
    path('therapist-sessions/', views.therapist_sessions, name='therapist_sessions'),
    path('therapist-session-details/', views.therapist_session_details, name='therapist_session_details'),
    path('therapist-analytics/', views.therapist_analytics, name='therapist_analytics'),

    path('live-session/', views.live_session, name='live_session'),
]