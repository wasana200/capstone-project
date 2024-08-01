from django.urls import path
from .views import upload_csv, csv_detail

urlpatterns = [
    path('', upload_csv, name='upload_csv'), 
    path('upload/', upload_csv, name='upload_csv'),
    path('csv/<int:pk>/', csv_detail, name='csv_detail'),
]
