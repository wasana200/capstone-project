# Generated by Django 5.0.7 on 2024-07-30 22:59

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='CSVFile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.FileField(upload_to='csvs/')),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
                ('prediction_result', models.TextField(blank=True, null=True)),
            ],
        ),
    ]
