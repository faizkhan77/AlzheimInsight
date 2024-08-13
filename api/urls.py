from django.urls import path, include
from . import views

urlpatterns = [
    path("predict/", views.predict_alzheimer, name="predict_alzheimers"),
]
