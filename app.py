from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Cargar el modelo y el escalador
modelo = pickle.load(open('modelo_svm_rbf.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    if request.method == 'POST':
        try:
            edad = int(request.form['edad'])
            tipo_empleado = int(request.form['tipo_empleado'])
            fnlwgt = int(request.form['fnlwgt'])
            educacion = int(request.form['educacion'])
            educacion_num = int(request.form['educacion_num'])
            estado_civil = int(request.form['estado_civil'])
            ocupacion = int(request.form['ocupacion'])
            relacion = int(request.form['relacion'])
            raza = int(request.form['raza'])
            sexo = int(request.form['sexo'])
            capital_ganado = int(request.form['capital_ganado'])
            capital_perdido = int(request.form['capital_perdido'])
            hr_por_semana = int(request.form['hr_por_semana'])
            pais = int(request.form['pais'])
            
            # Crear un array con los valores
            valores = np.array([[edad, tipo_empleado, fnlwgt, educacion, educacion_num, 
                                 estado_civil, ocupacion, relacion, raza, sexo, 
                                 capital_ganado, capital_perdido, hr_por_semana, pais]])
            
            # Escalar los valores
            valores_scaled = scaler.transform(valores)
            
            # Realizar la predicción
            prediccion = modelo.predict(valores_scaled)
            
            return render_template('resultado.html', prediccion=prediccion[0])
        except Exception as e:
            return f"Error en la predicción: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
