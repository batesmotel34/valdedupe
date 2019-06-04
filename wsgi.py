#! /usr/bin/env python3
import threading
from flask import Flask

# My typical setup for a Flask App.
# ./media is a folder that holds my JS, Imgs, CSS, etc.
app1 = Flask(__name__, static_folder='./media')
#app2 = Flask(__name__, static_folder='./media')

@app1.route('/')
def index1():
    return 'Hello World 1'

# @app2.route('/')
# def index2():
#     return 'Hello World 2'

# With Multi-Threading Apps, YOU CANNOT USE DEBUG!
# Though you can sub-thread.
def runFlaskApp1():
    app1.run(host='127.0.0.1', port=5000, debug=False, threaded=True)

# def runFlaskApp2():
#     app2.run(host='127.0.0.1', port=5001, debug=False, threaded=True)


if __name__ == '__main__':
    # Executing the Threads seperatly.
    t1 = threading.Thread(target=runFlaskApp1)
#     t2 = threading.Thread(target=runFlaskApp2)
    t1.start()
#     t2.start()
    