"""Create an AeroIntel account without printing or storing a token."""

import argparse
from getpass import getpass

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from source.auth import hash_password, normalize_email, valid_email
from source.database import SessionLocal
from source.models import User


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset-password",
        action="store_true",
        help="replace the password for an existing account",
    )
    args = parser.parse_args()

    email = normalize_email(input("Account email: "))
    if not valid_email(email):
        print("Enter a valid email address.")
        return 2

    password = getpass("Password (12–128 characters): ")
    confirmation = getpass("Confirm password: ")
    if password != confirmation:
        print("Passwords do not match.")
        return 2

    try:
        encoded_password = hash_password(password)
    except ValueError as error:
        print(str(error))
        return 2

    try:
        with SessionLocal() as db:
            existing = db.scalar(select(User).where(User.email == email))
            if existing is None and args.reset_password:
                print("No account with that email exists.")
                return 1
            if existing is not None and not args.reset_password:
                print("An account with that email already exists.")
                return 1

            if existing is not None:
                existing.password_hash = encoded_password
                existing.is_active = True
            else:
                db.add(User(email=email, password_hash=encoded_password))
            try:
                db.commit()
            except IntegrityError:
                db.rollback()
                print("An account with that email already exists.")
                return 1
    except Exception as error:
        print(
            "Account setup failed "
            f"(error type: {type(error).__name__}). "
            "Check the database connection and apply migrations."
        )
        return 1

    message = "Password reset." if args.reset_password else "Account created."
    print(f"{message} Sign in through Swagger at /docs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
