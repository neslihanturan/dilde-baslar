from django.urls import path
from . import views

urlpatterns = [
    path('process_input/', views.process_input, name='process_input'),
]

