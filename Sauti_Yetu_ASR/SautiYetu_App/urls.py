from django.urls import path, include
from . import views


urlpatterns = [    
    path('', views.home, name='home'),
    path('community/', views.community, name='community'),
    path('nearby/', views.nearby, name='nearby'),
    path('patient-dashboard/', views.patient_dashboard, name='patient_dashboard'),
    path('patient-session/', views.patient_sessions, name='patient_sessions'),
    path('patients/', views.patients, name='patients'),
    path('sessions/', views.sessions, name='session'),
    path('therapy-session/', views.therapy_session, name='therapist_session'),
    path('signin/', views.signin, name='signin'),
    path('signup/', views.signup, name='signup'),
]