"""User Management Page for Credit Scoring Dashboard.

This page is admin-only and provides:
- View all users
- Create new users
- Update user roles
- Deactivate users
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.auth import create_user
from backend.database import get_db_context
from backend.models import User, UserRole


def render_user_management():
    """Render the user management page."""
    # Check admin access
    from streamlit_app.auth import get_current_user

    user = get_current_user()
    if not user or user['role'] != UserRole.ADMIN.value:
        st.error("🔒 Access Denied: Admin privileges required")
        return

    # Create tabs
    tab1, tab2 = st.tabs(["👥 View Users", "➕ Create User"])

    with tab1:
        render_user_list()

    with tab2:
        render_create_user()


def render_user_list():
    """Render the user list."""
    st.markdown("### 👥 All Users")

    try:
        with get_db_context() as db:
            users = db.query(User).all()

            if users:
                # Convert to display format
                user_data = []
                for u in users:
                    user_data.append({
                        'ID': u.id,
                        'Username': u.username,
                        'Email': u.email or 'N/A',
                        'Role': u.role.value,
                        'Active': '✅ Yes' if u.is_active else '❌ No',
                        'Created': u.created_at.strftime('%Y-%m-%d %H:%M') if u.created_at else 'N/A'
                    })

                import pandas as pd
                df = pd.DataFrame(user_data)
                st.dataframe(df, use_container_width=True)

                st.write(f"**Total Users:** {len(users)}")

                # User actions
                st.markdown("---")
                st.subheader("User Actions")

                selected_user = st.selectbox(
                    "Select User",
                    options=[u.username for u in users],
                    help="Select a user to manage"
                )

                if selected_user:
                    col1, col2 = st.columns(2)

                    with col1:
                        new_role = st.selectbox(
                            "Change Role",
                            options=[r.value for r in UserRole],
                            help="Select new role for user"
                        )

                        if st.button("Update Role", use_container_width=True):
                            try:
                                with get_db_context() as db:
                                    user = db.query(User).filter(User.username == selected_user).first()
                                    if user:
                                        user.role = UserRole(new_role)
                                        db.commit()
                                        st.success(f"Updated role for {selected_user} to {new_role}")
                                        st.rerun()
                            except Exception as e:
                                st.error(f"Error updating role: {str(e)}")

                    with col2:
                        st.write("Deactivate User")

                        if st.button("🚫 Toggle Active Status", use_container_width=True):
                            try:
                                with get_db_context() as db:
                                    user = db.query(User).filter(User.username == selected_user).first()
                                    if user:
                                        user.is_active = not user.is_active
                                        db.commit()
                                        status = "activated" if user.is_active else "deactivated"
                                        st.success(f"User {selected_user} {status}")
                                        st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
            else:
                st.info("No users found in database.")

    except Exception as e:
        st.error(f"Error loading users: {str(e)}")


def render_create_user():
    """Render the create user form."""
    st.markdown("### ➕ Create New User")

    with st.form("create_user_form"):
        username = st.text_input(
            "Username",
            placeholder="Enter username",
            help="Unique username for the new user"
        )

        email = st.text_input(
            "Email",
            placeholder="user@example.com",
            help="User's email address"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter password",
            help="Password must be at least 8 characters"
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            placeholder="Confirm password"
        )

        role = st.selectbox(
            "Role",
            options=[r.value for r in UserRole],
            index=0,  # Default to ANALYST
            help="User role determines access level"
        )

        submitted = st.form_submit_button("Create User", use_container_width=True)

        if submitted:
            # Validation
            errors = []

            if not username:
                errors.append("Username is required")
            elif len(username) < 3:
                errors.append("Username must be at least 3 characters")

            if not password:
                errors.append("Password is required")
            elif len(password) < 8:
                errors.append("Password must be at least 8 characters")

            if password != confirm_password:
                errors.append("Passwords do not match")

            if errors:
                for error in errors:
                    st.error(error)
            else:
                try:
                    with get_db_context() as db:
                        # Check if username exists
                        existing = db.query(User).filter(User.username == username).first()
                        if existing:
                            st.error("Username already exists")
                        else:
                            # Create user
                            new_user = create_user(
                                db=db,
                                username=username,
                                email=email if email else None,
                                password=password,
                                role=UserRole(role)
                            )

                            if new_user:
                                st.success(f"✅ User '{username}' created successfully!")
                                st.info(f"User ID: {new_user.id}, Role: {role}")
                            else:
                                st.error("Failed to create user")

                except Exception as e:
                    st.error(f"Error creating user: {str(e)}")

    # Information box
    st.markdown("---")
    st.info("""
    **Role Descriptions:**
    - **ANALYST:** Can view predictions, run batch predictions, view model performance
    - **ADMIN:** Full access including monitoring dashboard and user management
    """)
