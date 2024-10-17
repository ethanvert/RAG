from flask import render_template, flash, redirect, url_for, request
from flask_login import current_user, login_user, logout_user, login_required
from backend import app, db
from backend.forms import LoginForm, RegistrationForm
from backend.dataclass import User
from urllib.parse import urlsplit

import sqlalchemy as sa


@app.route('/')
@app.route('/index')
@login_required
def index():
    """Renders the home page with user information and posts.
    
    Args:
        None
    
    Returns:
        str: Rendered HTML template for the home page.
    
    """    user = {'username': 'Ethan'}
    posts = [
        {
            'author': {'username': 'John'},
            'body': 'Beautiful day in Portland!'
        },
        {
            'author': {'username': 'Susan'},
            'body': 'The Avengers movie was so cool!'
        }
    ]
    return render_template("index.html", title='Home Page', posts=posts)


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Register a new user.
    
    This function handles the user registration process. It checks if the user is already
    authenticated, processes the registration form submission, creates a new user in the
    database, and redirects the user appropriately.
    
    Args:
        None
    
    Returns:
        flask.Response: A redirect response to either the index page (if user is already
        authenticated), the login page (upon successful registration), or the registration
        page with the form (if the form is invalid or not submitted).
    
    Raises:
        None
    
    Side effects:
        - Adds a new user to the database if registration is successful.
        - Flashes a success message upon successful registration.
    """    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login process.
    
    Args:
        None
    
    Returns:
        flask.Response: Redirects to the index page if the user is already authenticated,
        to the next page after successful login, or renders the login template for
        unsuccessful attempts.
    
    Raises:
        None
    
    Notes:
        This function manages the login process for users. It checks if the user
        is already authenticated, validates the login form, verifies user credentials,
        and handles the post-login redirection. Flash messages are used to communicate
        login failures to the user.
    """    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = db.session.scalar(
            sa.select(User).where(User.username == form.username.data))
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or urlsplit(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)


@app.route('/logout')
def logout():
    """Logs out the current user and redirects to the index page.
    
    This method performs two main actions:
    1. Logs out the currently authenticated user using the `logout_user()` function.
    2. Redirects the user to the index page of the application.
    
    Returns:
        redirect: A redirect response to the index page.
    """
    logout_user()
    return redirect(url_for('index'))


@app.route('/user/<username>')
@login_required
def user(username):
    """Render a user's profile page with their posts.
    
    Args:
        username (str): The username of the user whose profile is being requested.
    
    Returns:
        str: Rendered HTML template for the user's profile page.
    
    Raises:
        HTTPException: If the user is not found, a 404 error is raised by first_or_404().
    """    user = db.first_or_404(sa.select(User).where(User.username == username))
    posts = [
        {'author': user, 'body': 'Test post #1'},
        {'author': user, 'body': 'Test post #2'}
    ]
    return render_template('user.html', user=user, posts=posts)