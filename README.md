cd capstone-project

venv\Scripts\activate
source venv/bin/activate

pip install -r requirements.txt

python manage.py makemigrations
python3 manage.py migrate
python3 manage.py runserver

