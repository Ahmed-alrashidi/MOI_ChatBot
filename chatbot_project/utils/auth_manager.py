# =========================================================================
# File Name: utils/auth_manager.py
# Purpose: Interactive Secure User Authentication Management.
# Features:
# - Dynamic Pathing: Uses project root dynamically to ensure portability.
# - SHA-256 Hashing: Secures passwords without storing plain text.
# - CLI Interface: Provides an interactive English menu for user management.
# =========================================================================

import os
import json
import hashlib
import sys

# Ensure the project root is in the path to import Config properly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from config import Config

# Dynamically set the users directory relative to the project root
USERS_DIR = os.path.join(Config.PROJECT_ROOT, "users")
DB_FILE = os.path.join(USERS_DIR, "users_db.json")

def _hash_password(password: str, username: str = "") -> str:
    """
    Hashes the password with a username-based salt using SHA-256.
    The salt prevents rainbow table attacks on identical passwords.
    """
    salted = f"{username.lower().strip()}:{password}"
    return hashlib.sha256(salted.encode('utf-8')).hexdigest()

MIN_PASSWORD_LENGTH = 6

def init_db():
    """
    Initializes the user database directory and JSON file if they do not exist.
    """
    os.makedirs(USERS_DIR, exist_ok=True)
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, 'w') as f:
            json.dump({}, f)

def load_users() -> dict:
    """
    Loads the user data from the JSON database file.
    Returns an empty dictionary if the database is newly initialized.
    """
    init_db()
    with open(DB_FILE, 'r') as f:
        return json.load(f)

def save_users(users: dict):
    """
    Saves the user dictionary back to the JSON database file with formatting.
    """
    with open(DB_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def add_user_interactive():
    """
    CLI interface to add a new user. Prompts for ID and password,
    hashes the password, and saves the record safely.
    """
    print("\n--- ➕ Add New User ---")
    username = input("Enter User ID: ").strip()
    if not username:
        print("⚠️ Error: User ID cannot be empty.")
        return
        
    password = input("Enter Password: ").strip()
    if not password:
        print("⚠️ Error: Password cannot be empty.")
        return
    if len(password) < MIN_PASSWORD_LENGTH:
        print(f"⚠️ Error: Password must be at least {MIN_PASSWORD_LENGTH} characters.")
        return
        
    users = load_users()
    if username in users:
        print(f"⚠️ User ({username}) already exists in the system!")
    else:
        users[username] = _hash_password(password, username)
        save_users(users)
        print(f"✅ User ({username}) added successfully and securely.")

def delete_user_interactive():
    """
    CLI interface to delete an existing user. 
    Requires explicit confirmation before deletion to prevent accidental data loss.
    """
    print("\n--- 🗑️ Delete User ---")
    username = input("Enter User ID to delete: ").strip()
    
    users = load_users()
    if username in users:
        # Require explicit confirmation before executing deletion
        confirm = input(f"Are you sure you want to delete user ({username})? (y/n): ").strip().lower()
        if confirm == 'y':
            del users[username]
            save_users(users)
            print(f"✅ User ({username}) deleted successfully.")
        else:
            print("❌ Deletion cancelled.")
    else:
        print(f"⚠️ Error: User ({username}) not found in the database.")

def view_users_interactive():
    """
    CLI interface to display all authorized users.
    Hides the hashed passwords for security purposes.
    """
    users = load_users()
    print("\n--- 👥 Authorized Users List ---")
    if not users:
        print("📭 No users currently registered.")
    else:
        print("-" * 40)
        for idx, username in enumerate(users.keys(), 1):
            # Display only the username/ID, keeping the hash strictly hidden
            print(f"{idx}. User ID: {username} | (Password: Hashed 🔒)")
        print("-" * 40)
        print(f"Total Users: {len(users)}")

def verify_user(username: str, password: str) -> bool:
    """
    Silent authentication function used by the Gradio interface in main.py.
    Supports both salted (new) and unsalted (legacy) password hashes.
    """
    users = load_users()
    if username not in users:
        return False
    stored_hash = users[username]
    # Try salted hash first (new format)
    if stored_hash == _hash_password(password, username):
        return True
    # Fallback: try unsalted hash (legacy users created before salt was added)
    legacy_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
    if stored_hash == legacy_hash:
        # Auto-migrate to salted hash
        users[username] = _hash_password(password, username)
        save_users(users)
        return True
    return False

def change_password_interactive():
    """
    CLI interface to change an existing user's password.
    Requires the old password for verification before allowing the change.
    """
    print("\n--- 🔑 Change Password ---")
    username = input("Enter User ID: ").strip()
    
    users = load_users()
    if username not in users:
        print(f"⚠️ Error: User ({username}) not found.")
        return
    
    old_password = input("Enter current password: ").strip()
    if not verify_user(username, old_password):
        print("❌ Error: Current password is incorrect.")
        return
    
    new_password = input("Enter new password: ").strip()
    if len(new_password) < MIN_PASSWORD_LENGTH:
        print(f"⚠️ Error: Password must be at least {MIN_PASSWORD_LENGTH} characters.")
        return
    
    users = load_users()  # Reload in case verify_user migrated the hash
    users[username] = _hash_password(new_password, username)
    save_users(users)
    print(f"✅ Password changed successfully for ({username}).")

def main_menu():
    """
    The main interactive Command Line Interface (CLI) loop for system administrators.
    """
    while True:
        print("\n" + "="*45)
        print("🛡️  Absher Smart Assistant Auth Manager  🛡️")
        print("="*45)
        print("1. ➕ Add New User")
        print("2. 🗑️  Delete User")
        print("3. 🔑 Change Password")
        print("4. 👥 View Authorized Users")
        print("5. 🚪 Exit System")
        print("="*45)
        
        choice = input("Select an action (1-5): ").strip()
        
        if choice == '1':
            add_user_interactive()
        elif choice == '2':
            delete_user_interactive()
        elif choice == '3':
            change_password_interactive()
        elif choice == '4':
            view_users_interactive()
        elif choice == '5':
            print("\n👋 Exiting management system. Goodbye!\n")
            break
        else:
            print("⚠️ Invalid choice, please enter a number from 1 to 5.")

if __name__ == "__main__":
    # Launch the interactive menu when the script is executed directly
    main_menu()