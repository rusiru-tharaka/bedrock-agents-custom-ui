import json
import os
from services import bedrock_agent_runtime
import streamlit as st
import uuid

# Get config from environment variablese
agent_id = os.environ.get("BEDROCK_AGENT_ID")
agent_alias_id = os.environ.get("BEDROCK_AGENT_ALIAS_ID", "StratDocAssist") # TSTALIASID is the default test alias ID
ui_title = os.environ.get("BEDROCK_AGENT_TEST_UI_TITLE", "OCSO - DocAi Assistant")
#ui_subheader = st.subheader("_Streamlit_ is :blue[cool] :sunglasses:")
ui_icon = os.environ.get("BEDROCK_AGENT_TEST_UI_ICON")

def init_state():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.citations = []
    st.session_state.trace = {}

# General page configuration and initialization
st.set_page_config(page_title=ui_title, page_icon=ui_icon, layout="wide")
st.title(ui_title)
if len(st.session_state.items()) == 0:
    init_state()

# Sidebar Logo
LOGO = "resource/images/logo-1.png"
st.sidebar.image(LOGO)
st.sidebar.caption("Introducing StratDocAssist, the cutting-edge artificial intelligence solution from Aboitiz Power, designed to revolutionize document analysis and empower businesses with unparalleled efficiency.")

# Sidebar button to reset session state
with st.sidebar:
    if st.button("Reset Session"):
        init_state()

# Messages in the conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Chat input that invokes the agent
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("...")
        response = bedrock_agent_runtime.invoke_agent(
            agent_id,
            agent_alias_id,
            st.session_state.session_id,
            prompt
        )
        output_text = response["output_text"]

        # Add citations
        if len(response["citations"]) > 0:
            citation_num = 1
            num_citation_chars = 0
            citation_locs = ""
            for citation in response["citations"]:
                end_span = citation["generatedResponsePart"]["textResponsePart"]["span"]["end"] + 1
                for retrieved_ref in citation["retrievedReferences"]:
                    citation_marker = f"[{citation_num}]"
                    output_text = output_text[:end_span + num_citation_chars] + citation_marker + output_text[end_span + num_citation_chars:]
                    citation_locs = citation_locs + "\n<br>" + citation_marker + " " + retrieved_ref["location"]["s3Location"]["uri"]
                    citation_num = citation_num + 1
                    num_citation_chars = num_citation_chars + len(citation_marker)
                output_text = output_text[:end_span + num_citation_chars] + "\n" + output_text[end_span + num_citation_chars:]
                num_citation_chars = num_citation_chars + 1
            output_text = output_text + "\n" + citation_locs

        placeholder.markdown(output_text, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": output_text})
        st.session_state.citations = response["citations"]
        st.session_state.trace = response["trace"]

trace_type_headers = {
    "preProcessingTrace": "Pre-Processing",
    "orchestrationTrace": "Orchestration",
    "postProcessingTrace": "Post-Processing"
}
trace_info_types = ["invocationInput", "modelInvocationInput", "modelInvocationOutput", "observation", "rationale"]


    