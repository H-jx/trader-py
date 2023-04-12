

import threading

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from trader import analyze_data

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('message')
def handle_message(message):
    print('Received message: ' + message)
    emit('message', message, broadcast=True)

if __name__ == '__main__':
    t = threading.Thread(target=analyze_data)
    t.start()
    socketio.run(app)
