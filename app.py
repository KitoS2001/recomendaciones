from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import random
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Generar datos de ejemplo para las transacciones y el catálogo
products = [
    'audifonos', 'impresoras', 'memorias', 'proyectores', 'smartwatch', 
    'adaptadores', 'laptops', 'tablets', 'camaras', 'smartphones', 
    'televisores', 'consolas', 'routers', 'monitores', 'teclados',
    'ratones', 'altavoces', 'microfonos', 'discos_duros', 'ssd', 
    'lectores_de_tarjetas', 'cables_hdmi', 'cables_usb', 'protectores_de_pantalla',
]

# Generar transacciones de ejemplo
def generate_transactional_data(num_transactions, products):
    data = []
    for _ in range(num_transactions):
        transaction = random.sample(products, k=random.randint(1, 5))
        data.append(transaction)
    return data

# Crear el DataFrame de transacciones
num_transactions = 1000
transactions = generate_transactional_data(num_transactions, products)
df = pd.DataFrame(transactions)

# Codificar las transacciones en un formato binario
df_encoded = df.apply(lambda x: pd.Series(1, index=x.dropna()), axis=1).fillna(0).astype(bool)

# Aplicar el algoritmo Apriori
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

# Generar reglas de asociación
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/catalog')
def catalog():
    return render_template('catalog.html', products=products)

@app.route('/buy/<product>')
def buy(product):
    if 'cart' not in session:
        session['cart'] = []
    session['cart'].append(product)

    # Obtener recomendaciones basadas en las compras
    recommendations = recommend_products(session['cart'])
    
    return render_template('recommendations.html', product=product, recommendations=recommendations)

def recommend_products(cart):
    recommended = set()

    for item in cart:
        related_rules = rules[rules['antecedents'].apply(lambda x: item in x)]
        for _, row in related_rules.iterrows():
            for consequent in row['consequents']:
                if consequent not in cart:
                    recommended.add(consequent)
    
    return list(recommended)

if __name__ == '__main__':
    app.run(debug=True)
