from flask import Flask, render_template, request, redirect, url_for, jsonify,flash
import os
from werkzeug.utils import secure_filename
from detect_fraud import detect_fraud  # Le module qui appelle YOLO
from bulletin import process_document
from extract import process_image
from ccn import predict_image_class
from num import detect_and_save , extract_bulletin_numbers
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask import session

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
PROCESSED_FOLDER = 'processed'


app = Flask(__name__)
app.secret_key = 'flasknarjess'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///documents.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Modèles SQLAlchemy
class Ordonnance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ocr_text = db.Column(db.Text)
    doctor_name = db.Column(db.String(100))
    time = db.Column(db.String(50))
    image = db.Column(db.String(150))

class Bulletin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    bulletin_number = db.Column(db.String(50))
    time = db.Column(db.String(50))
    image_initial = db.Column(db.String(150))
    image_final = db.Column(db.String(150))

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False)
    password = db.Column(db.String(200), nullable=False)
    phone = db.Column(db.String(15), nullable=False)
    profession = db.Column(db.String(100), nullable=False)
    organisation = db.Column(db.String(100), nullable=False)

# Création DB si pas existante
with app.app_context():
    db.create_all()


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Recherche de l'utilisateur dans la base
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id  # Garde l'utilisateur connecté
            flash('Connexion réussie !', 'success')
            return redirect(url_for('home'))  # Redirige vers un dashboard ou page d'accueil
        else:
            flash('Nom d’utilisateur ou mot de passe incorrect', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')




@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        phone = request.form['phone']
        profession = request.form['profession']
        organisation = request.form['organisation']

        # Vérification basique côté serveur
        if len(username) < 4 or len(password) < 8:
            flash("Erreur: Données invalides", "error")
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)

        user = User(
            username=username,
            password=hashed_password,
            phone=phone,
            profession=profession,
            organisation=organisation
        )

        db.session.add(user)
        db.session.commit()

        flash("Inscription réussie !", "success")
        return redirect(url_for('login'))  # ou la route de ton choix

    return render_template('register.html')  # ton formulaire HTML


@app.route('/users')
def list_users():
    users = User.query.all()
    return "<br>".join([f"{u.id} | {u.username} | {u.phone} | {u.profession} | {u.organisation}" for u in users])


from flask import send_from_directory, url_for

def encode_image_to_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Vérifie si le fichier a une extension autorisée
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/fraud')
def fraud():
    return render_template('fraud.html')
'''
@app.route('/extract', methods=['GET', 'POST'])
def prescription():
    doctor_text = ""
    ocr_text = ""
    image_url = ""

    if request.method == 'POST':
        image = request.files['image']
        if image:
            image_path = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(image_path)

            # Fichiers de sortie
            doctor_path = os.path.join(PROCESSED_FOLDER, "nom_docteur.txt")
            ocr_path = os.path.join(PROCESSED_FOLDER, "textes_ordonnes.txt")

            # Traitement principal
            process_image(image_path, doctor_path, ocr_path)

            # Lire les résultats
            doctor_text = open(doctor_path, encoding="utf-8").read() if os.path.exists(doctor_path) else "Non détecté"
            ocr_text = open(ocr_path, encoding="utf-8").read() if os.path.exists(ocr_path) else "Aucun texte détecté"
            image_url = url_for('uploaded_file', filename=image.filename)

    return render_template('prescrition.html',
                           image_url=image_url,
                           doctor_name=doctor_text,
                           ocr_text=ocr_text)




'''

from werkzeug.utils import secure_filename
'''
@app.route('/extract', methods=['GET', 'POST'])
def extract_view():
    if request.method == 'POST':
        uploaded_file = request.files['image']
        if uploaded_file:
            filename = secure_filename(uploaded_file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(path)
            results = process_document(path)
            return render_template("extract3.html", image_url=results["image_with_text"], original=filename)
    return render_template("extract3.html")


@app.route('/extract', methods=['GET', 'POST'])
def extract_all():
    if request.method == 'POST':
        uploaded_file = request.files['image']
        if uploaded_file:
            filename = secure_filename(uploaded_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(image_path)

            # Étape 1 : prédire le type du document
            document_type = predict_image_class(image_path)  # retourne "prescription" ou "bulletin"

            if document_type == "Ordonnaces":
                # Traitement spécifique à l'ordonnance
                doctor_path = os.path.join(PROCESSED_FOLDER, "nom_docteur.txt")
                ocr_path = os.path.join(PROCESSED_FOLDER, "textes_ordonnes.txt")
                process_image(image_path, doctor_path, ocr_path)

                doctor_text = open(doctor_path, encoding="utf-8").read() if os.path.exists(doctor_path) else "Non détecté"
                ocr_text = open(ocr_path, encoding="utf-8").read() if os.path.exists(ocr_path) else "Non détecté"
                return render_template("prescrition.html", doctor_name=doctor_text, ocr_text=ocr_text, image_url=url_for('uploaded_file', filename=filename))

            else:
                # Traitement générique (bulletin)
                results = process_document(image_path)  # retourne un dictionnaire avec 'image_with_text'

                return render_template("extract3.html", image_url=results["image_with_text"], original=filename)

    return render_template("extract3.html")  # affichage de base si GET
'''


@app.route('/extract', methods=['GET', 'POST']) 
def extract_all():
    if request.method == 'POST':
        uploaded_file = request.files['image']
        if uploaded_file:
            filename = secure_filename(uploaded_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(image_path)

            document_type = predict_image_class(image_path)
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            if document_type == "Ordonnaces":
                doctor_path = os.path.join('processed', "nom_docteur.txt")
                ocr_path = os.path.join('processed', "textes_ordonnes.txt")
                process_image(image_path, doctor_path, ocr_path)

                doctor_text = open(doctor_path, encoding="utf-8").read() if os.path.exists(doctor_path) else "Non détecté"
                ocr_text = open(ocr_path, encoding="utf-8").read() if os.path.exists(ocr_path) else "Non détecté"

                # Sauvegarde dans SQLite
                ordonnance = Ordonnance(
                    ocr_text=ocr_text,
                    doctor_name=doctor_text,
                    time=now,
                    image=filename
                )
                db.session.add(ordonnance)
                db.session.commit()

                return render_template("prescription.html", doctor_name=doctor_text, ocr_text=ocr_text, image_url=url_for('uploaded_file', filename=filename))

            else:
                result = process_document(image_path)
                final_image_name = result["image_with_text"]
                detection_txt = result["text_path"]

                bulletin_nums = extract_bulletin_numbers(detection_txt, image_path)
                bulletin_id = bulletin_nums[0] if bulletin_nums else "Non détecté"

                # Sauvegarde dans SQLite
                bulletin = Bulletin(
                    bulletin_number=bulletin_id,
                    time=now,
                    image_initial=filename,
                    image_final=result["image_with_text"]    #final_image_name
                )
                db.session.add(bulletin)
                db.session.commit()

                return render_template("extract5.html", image_url=final_image_name, original=filename)

    return render_template("extract5.html")


# Route pour visualiser les ordonnances
@app.route('/ordonnances')
def view_ordonnances():
    ordos = Ordonnance.query.all()
    return render_template("ordonnance.html", ordonnances=ordos)

# Route pour visualiser les bulletins
@app.route('/bulletins')
def view_bulletins():
    bulletins = Bulletin.query.all()
    return render_template("bulletin.html", bulletins=bulletins)


'''
@app.route('/extract', methods=['GET', 'POST'])
def extract_all():
    if request.method == 'POST':
        uploaded_file = request.files['image']
        if uploaded_file:
            filename = secure_filename(uploaded_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(image_path)

            # Étape 1 : prédire le type du document
            document_type = predict_image_class(image_path)  # retourne "prescription" ou "bulletin"

            if document_type == "Ordonnaces":
                # Traitement spécifique à l'ordonnance
                doctor_path = os.path.join(PROCESSED_FOLDER, "nom_docteur.txt")
                ocr_path = os.path.join(PROCESSED_FOLDER, "textes_ordonnes.txt")
                process_image(image_path, doctor_path, ocr_path)

                doctor_text = open(doctor_path, encoding="utf-8").read() if os.path.exists(doctor_path) else "Non détecté"
                ocr_text = open(ocr_path, encoding="utf-8").read() if os.path.exists(ocr_path) else "Non détecté"
                return render_template("prescrition.html", doctor_name=doctor_text, ocr_text=ocr_text, image_url=url_for('uploaded_file', filename=filename))

            else:
                # Traitement générique (bulletin)
                results = process_document(image_path)  # retourne un dictionnaire avec 'image_with_text'

                return render_template("extract3.html", image_url=results["image_with_text"], original=filename)

    return render_template("extract3.html")  # affichage de base si GET
'''

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'result': "Erreur : aucune image fournie."})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'result': "Erreur : nom de fichier vide."})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        # Appelle ton module de détection ici
        label_path = filepath.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
        try:
            result = detect_fraud(filepath, label_path)
        except Exception as e:
            result = f"Error during detection: {str(e)}"

        return jsonify({'result': result})  # ← retourne JSON ici

    return jsonify({'result': "Error: Unsupported file format."})

if __name__ == '__main__':
    app.run(debug=True)
    result = predict_image_class("uploads/0711--9459343--20230731_page_0.jpg")
    print("Classe prédite :", result)
