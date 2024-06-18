from flask import Flask, render_template

app = Flask(__name__, template_folder='/Users/ethanvertal/Documents/RAG Site/frontend/templates')


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)