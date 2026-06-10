import streamlit as st
from auth import register_user, login_user, generate_token

def show_login_page():
    """
    Show login/signup UI.
    Returns True if user is authenticated, False otherwise.
    """
    st.title("💚 Mental Health Companion")
    st.caption("A safe, private space to express how you feel.")

    st.markdown("---")

    tab_login, tab_signup = st.tabs(["🔑 Login", "✨ Sign Up"])

    with tab_login:
        st.subheader("Welcome back!")
        with st.form("login_form"):
            email    = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password")
            submit   = st.form_submit_button(
                "Login", use_container_width=True, type="primary"
            )

        if submit:
            if not email or not password:
                st.error("Please enter both email and password.")
            else:
                success, result = login_user(email, password)
                if success:
                    token = generate_token(result["id"], result["email"])
                    st.session_state.token   = token
                    st.session_state.user_id = result["id"]
                    st.session_state.user_name  = result["name"]
                    st.session_state.user_email = result["email"]
                    st.success(f"Welcome back, {result['name']}! 💚")
                    st.rerun()
                else:
                    st.error(result)

    with tab_signup:
        st.subheader("Create your account")
        st.info("Your journal is completely private — only you can see it.")
        with st.form("signup_form"):
            name     = st.text_input("Your name", placeholder="Sreeja")
            email    = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input(
                "Password", type="password",
                help="At least 6 characters"
            )
            confirm  = st.text_input("Confirm password", type="password")
            submit   = st.form_submit_button(
                "Create Account", use_container_width=True, type="primary"
            )

        if submit:
            if not name or not email or not password:
                st.error("Please fill in all fields.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters.")
            elif password != confirm:
                st.error("Passwords don't match.")
            else:
                success, result = register_user(email, password, name)
                if success:
                    token = generate_token(result, email)
                    st.session_state.token      = token
                    st.session_state.user_id    = result
                    st.session_state.user_name  = name
                    st.session_state.user_email = email
                    st.success(f"Account created! Welcome, {name}! 💚")
                    st.rerun()
                else:
                    st.error(result)

    return False

def is_authenticated():
    """Check if user has valid session."""
    from auth import verify_token
    if "token" not in st.session_state:
        return False
    valid, payload = verify_token(st.session_state.token)
    if not valid:
        # Clear invalid session
        for key in ["token","user_id","user_name","user_email"]:
            st.session_state.pop(key, None)
        return False
    return True