from django.shortcuts import render, HttpResponse

# Create your views here.


def index(request):
    return render(request, "index.html")


def modelform(request):
    return render(request, "modelform.html")
