from django.shortcuts import render, HttpResponse
from django.views.decorators.csrf import csrf_exempt

# Create your views here.


def index(request):
    return render(request, "index.html")


@csrf_exempt
def modelform(request):
    return render(request, "modelform.html")
