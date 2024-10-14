from backend import login
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timezone
from typing import Optional
import sqlalchemy as sa
import sqlalchemy.orm as so
from backend import db
from flask_login import UserMixin
from hashlib import md5


@login.user_loader
def load_user(id):
    """Load a user from the database by their ID.
    
    Args:
        id (int): The unique identifier of the user to load.
    
    Returns:
        User: The User object corresponding to the given ID.
    
    Raises:
        ValueError: If the provided ID cannot be converted to an integer.
        SQLAlchemyError: If there's an issue with the database connection or query.
    """    return db.session.get(User, int(id))


class User(UserMixin, db.Model):
    id: so.Mapped[int] = so.mapped_column(primary_key=True)
    username: so.Mapped[str] = so.mapped_column(sa.String(64), index=True,
                                                unique=True)
    email: so.Mapped[str] = so.mapped_column(sa.String(120), index=True,
                                             unique=True)
    password_hash: so.Mapped[Optional[str]] = so.mapped_column(sa.String(256))
    about_me: so.Mapped[Optional[str]] = so.mapped_column(sa.String(140))
    last_seen: so.Mapped[Optional[datetime]] = so.mapped_column(
        default=lambda: datetime.now(timezone.utc))

    def set_password(self, password) -> None:
        """Sets the password for the user by generating and storing a password hash.
        
        Args:
            password (str): The plain text password to be hashed and stored.
        
        Returns:
            None: This method doesn't return anything.
        """        self.password_hash = generate_password_hash(password)

    def check_password(self, password) -> bool:
        """Checks if the provided password matches the stored password hash.
        
        Args:
            password (str): The password to be verified.
        
        Returns:
            bool: True if the password matches the stored hash, False otherwise.
        """        return check_password_hash(self.password_hash, password)

    def __repr__(self) -> str:
        """Returns a string representation of the User object.
        
        Args:
            self: The User instance.
        
        Returns:
            str: A string representation of the User object in the format '<User {username}>'.
        """        return '<User {}>'.format(self.username)

    def avatar(self, size):
        """Generate a Gravatar URL for the user's email address.
        
        Args:
            size (int): The size of the Gravatar image in pixels.
        
        Returns:
            str: A URL to the Gravatar image for the user's email address.
        """        digest = md5(self.email.lower().encode('utf-8')).hexdigest()
        return f'https://www.gravatar.com/avatar/{digest}?d=identicon&s={size}'


class Post(db.Model):
    id: so.Mapped[int] = so.mapped_column(primary_key=True)
    body: so.Mapped[str] = so.mapped_column(sa.String(140))
    timestamp: so.Mapped[datetime] = so.mapped_column(
        index=True, default=lambda: datetime.now(timezone.utc))
    user_id: so.Mapped[int] = so.mapped_column(sa.ForeignKey(User.id),
                                               index=True)

    author: so.Mapped[User] = so.relationship(back_populates='posts')

    def __repr__(self):
        """Returns a string representation of the Post object.
        
        Args:
            self: The instance of the Post class.
        
        Returns:
            str: A string representation of the Post object, containing the body of the post.
        """        return '<Post {}>'.format(self.body)