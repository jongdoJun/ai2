from flask import Flask
from flask import render_template
from machine.random_number_maker import RandomNumberMaker
from machine.boston import Boston
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move/<path>')
def move(path):
    return render_template(f'{path}.html')

@app.route('/random_number')
def random_number():
    rnm = RandomNumberMaker()
    result = rnm.fetch_random_number()
    render_params = {}
    render_params['result'] = result
    return render_template('random_number.html', **render_params)

@app.route('/neuron')
def neuron():
    rnm = RandomNumberMaker()
    result = rnm.create_neuron()
    render_params = {}
    render_params['result'] = result
    return render_template('nueron.html', **render_params)

@app.route('/boston')
def boston():
    boston = Boston()
    boston.create_model()

    render_params = {}
    render_params['result'] = None
    return render_template('boston.html', **render_params)

if __name__ == '__main__':
    app.run()