from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import detect_sexism

@csrf_exempt
def process_input(request):
    if request.method == 'POST':
        input_data = request.POST.get('input_data')
        print("this "+input_data)
        processed_data = detect_sexism(input_data)   # Call the method to check for sexism
        return JsonResponse({'processed_data': processed_data})
    else:
        return JsonResponse({'error': 'Only POST requests are accepted.'})

