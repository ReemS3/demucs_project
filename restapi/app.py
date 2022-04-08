from flask import Flask, render_template, request
import os
import joblib
# from model.separate import separate_music

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'uploads')


@app.route('/',  methods=['GET', 'POST'])
def home():
    print(request.form)
    return render_template('home.html')


@app.route('/separate', methods=['GET', 'POST'])
def separate():
    if request.method == 'POST':
        song = int(request.form.get('comment'))
        if song == 1:
            audio = "lush"
        elif song == 2:
            audio = "run"
        elif song == 3:
            audio = "talk"
        elif song == 4:
            audio = "shore"
        elif song == 5:
            audio = "fire"
    return render_template('separation_results.html', audio=audio)


if __name__ == '__main__':
    app.run()
