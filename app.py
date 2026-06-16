import streamlit as st
import os
import agent_controller as agent_controller

st.set_page_config(page_title="Smart Data Mining Agent", layout="wide")
st.title("Smart Data Mining Agent with LangChain and Metis AI")
st.write("Enter your request (e.g., Load my data, preprocess it, and build a decision tree model)")

# Upload section in sidebar
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Store file_path in session_state so the agent can always access it
if "file_path" not in st.session_state:
    st.session_state.file_path = None

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state.file_path = file_path
    st.sidebar.success(f"File saved successfully")

# Initialize the current dataframe variable
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize the agent executor if not already done
current_path = st.session_state.file_path or ""
if "executor" not in st.session_state or st.session_state.get("executor_path") != current_path:
    st.session_state.executor = agent_controller.build_executor(current_path)
    st.session_state.executor_path = current_path

# Get the agent executor
if user_query := st.chat_input("Type your request here..."):
    # Display the user's query in the chat interface
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Execute the agent's response based on the user's query
    with st.chat_message("assistant"):
        with st.spinner("Processing your request..."):
            try:
                # Invoke the agent with the user's query and get the response
                response = st.session_state.executor.invoke({
                    "input": user_query,
                })
                output_text = response["output"]
                st.markdown(output_text)

                # Show download button if a report was generated
                report_path = agent_controller.agent_state.get("report_path")
                if report_path and os.path.exists(report_path):
                    with open(report_path, "rb") as rf:
                        st.download_button(
                            label="📄 Download Report",
                            data=rf,
                            file_name=os.path.basename(report_path),
                            mime="text/plain",
                        )

            except Exception as e:
                output_text = f"Error occurred while processing the request: {str(e)}"
                st.error(output_text)
                
    st.session_state.messages.append({"role": "assistant", "content": output_text})