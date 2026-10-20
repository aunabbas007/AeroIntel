
from source.auth import create_access_token

# Local development only — this is a fixed test user.
token = create_access_token("aerointel_user")

print("\nYour development access token:\n")
print(token)
print("\nToken expires in 30 minutes.")
