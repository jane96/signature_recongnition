from flask import render_template
from app import app

@app.route('/')
@app.route('/first')
def first():
   return render_template("first.html",title = 'Home')
@app.route('/second')
def second():
    user = { 'nickname': 'Miguel' } # fake user
    return render_template("second.html",
        title = 'Home',
        user = user)
@app.route('/thrid')
def thrid():
    user = { 'nickname': 'Miguel' } # fake user
    return render_template("thrid.html",
        title = 'Home',
        user = user)
@app.route('/four')
def four():
    user = { 'nickname': 'Miguel' } # fake user
    return render_template("four.html",
        title = 'Home',
        user = user)
@app.route('/five')
def five():
    user = { 'nickname': 'Miguel' } # fake user
    return render_template("five.html",
        title = 'Home',
        user = user)




